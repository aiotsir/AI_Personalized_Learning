import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from unsloth import FastLanguageModel # Correct import for the Socratic Tutor
from huggingface_hub import hf_hub_download # For loading the GRU files

# --- Page Configuration ---
st.set_page_config(page_title="EULER - Socratic AI Tutor", layout="wide")

# --- Question Bank (No Changes) ---
QUESTION_BANK = [
    {"id": "q1", "latex": r'''\text{1. A rectangle has a length of } 8 \text{ cm and a width of } 5 \text{ cm. What is its area?}''', "answer": 40, "skill": "area", "problemType": "algebra", "hint": "The area of a rectangle is calculated by multiplying its length by its width."},
    {"id": "q2", "latex": r'''\text{2. If a train travels at } 60 \text{ km/h, how far will it travel in } 3 \text{ hours?}''', "answer": 180, "skill": "distance-rate-time", "problemType": "word-problem", "hint": "Distance = Speed × Time."},
    {"id": "q3", "latex": r'''\text{3. What is the value of x in the equation } 2x + 5 = 15?''', "answer": 5, "skill": "solving-linear-equations", "problemType": "algebra", "hint": "First, isolate the term with 'x' by subtracting 5 from both sides."},
    {"id": "q4", "latex": r'''\text{4. A pizza is cut into 8 equal slices. If you eat 3 slices, what fraction is left? (Answer as a decimal)}''', "answer": 0.625, "skill": "fractions", "problemType": "word-problem", "hint": "If you eat 3 slices, how many are left out of the original 8? Convert that fraction to a decimal."},
    {"id": "q5", "latex": r'''\text{5. What is } 15\% \text{ of } 200?''', "answer": 30, "skill": "percentages", "problemType": "arithmetic", "hint": "To find the percentage, you can multiply 200 by 0.15."}
]

# --- GRU Model Definition (No Changes) ---
class LearnerProfilerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LearnerProfilerGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

# --- EDITED SECTION: Loading All Models and Artifacts from Hugging Face Hub ---
@st.cache_resource(show_spinner="Warming up the AI engines...")
def load_all_models_and_artifacts():
    device = torch.device('cpu')
    # This is your public repository where all model files are stored.
    HF_REPO_ID = "Hfaceto/Bris_Socratic_Tutor"

    # --- Load Learner Profiler (GRU) from Hub ---
    profiler_model, scalers, mappings = None, None, None
    try:
        # Download each file individually from your Hub repository
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="learner_profiler_gru.pth")
        scalers_path = hf_hub_download(repo_id=HF_REPO_ID, filename="scalers.pkl")
        mappings_path = hf_hub_download(repo_id=HF_REPO_ID, filename="mappings.pkl")

        with open(scalers_path, 'rb') as f: scalers = pickle.load(f)
        with open(mappings_path, 'rb') as f: mappings = pickle.load(f)

        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 8, 64, 4
        profiler_model = LearnerProfilerGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        profiler_model.load_state_dict(torch.load(model_path, map_location=device))
        profiler_model.eval()
    except Exception as e:
        st.error(f"Error loading Learner Profiler from Hub: {e}. Ensure files exist in the repo '{HF_REPO_ID}'.")

    # --- Load Socratic Tutor (Fine-tuned Phi-3 with Unsloth) from Hub ---
    socratic_model, socratic_tokenizer = None, None
    try:
        socratic_model, socratic_tokenizer = FastLanguageModel.from_pretrained(
            model_name=HF_REPO_ID, # Load the entire repo
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(socratic_model)
    except Exception as e:
        st.error(f"Error loading Socratic Tutor from Hub: {e}. Ensure all adapter and tokenizer files are present.")

    return profiler_model, scalers, mappings, socratic_model, socratic_tokenizer

# --- Inference Functions (Your Logic Preserved) ---
def predict_student_state(sequence_df, model, scalers, mappings):
    if model is None or sequence_df.empty: return pd.DataFrame()
    feature_order = ['skill_encoded', 'problemType_encoded', 'correct', 'attemptCount', 'totalHints', 'log_timeTaken', 'time_per_attempt', 'prev_correct']
    processed_df = sequence_df.copy()
    processed_df['skill_encoded'] = processed_df['skill'].apply(lambda x: mappings['skill'].get(x, -1))
    processed_df['problemType_encoded'] = processed_df['problemType'].apply(lambda x: mappings['problemType'].get(x, -1))
    for col in feature_order:
        if col in scalers:
            processed_df[col] = scalers[col].transform(processed_df[[col]])
    feature_tensor = torch.tensor(processed_df[feature_order].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(feature_tensor, [len(processed_df)])
    prediction_np = prediction.squeeze(0).cpu().numpy()
    output_states = ['CONCENTRATING', 'GAMING', 'OFF TASK', 'BORED']
    result_df = pd.DataFrame(prediction_np, columns=output_states)
    result_df['Dominant State'] = result_df.idxmax(axis=1)
    return result_df

def get_socratic_feedback(tutor_model, tokenizer, dominant_state, question_text, user_answer, is_correct):
    if tutor_model is None: return "Error: Tutor model not loaded."
    
    if is_correct:
        prompt_content = f"My student's dominant state is CONCENTRATING and they answered correctly.\nProblem: '{question_text}'\nCorrect Answer: {user_answer}\nEngage them with an enrichment question."
    else:
        prompt_content = f"The student's dominant state is {dominant_state}. They are on the problem '{question_text}' and their incorrect answer was {user_answer}. Guide them Socratically."
    
    messages = [{"role": "user", "content": prompt_content}]
    # Use CPU for generation on Streamlit Cloud
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = tutor_model.generate(
            input_ids=input_ids, max_new_tokens=60, use_cache=True,
            do_sample=True, temperature=0.6, top_p=0.9
        )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response.split("<|assistant|>")[-1].strip()

# --- User Authentication (No Changes) ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
authenticator.login()

# --- Main Application Logic (Your Structure Preserved) ---
if st.session_state.get("authentication_status"):
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome {st.session_state.get('name')}")
    profiler_model, scalers, mappings, socratic_model, socratic_tokenizer = load_all_models_and_artifacts()

    # Stop the app if models fail to load
    if profiler_model is None or socratic_model is None:
        st.error("A critical AI model failed to load. The application cannot continue. Please check the logs.")
        st.stop()

    # --- Session State Initialization ---
    def initialize_quiz():
        st.session_state.session_sequence = []
        st.session_state.prev_correct = 0
        st.session_state.last_action_time = time.time()
        st.session_state.problem_states = {q['id']: {'attempts': 0, 'hints': 0} for q in QUESTION_BANK}
        st.session_state.socratic_dialogue = [{"role": "assistant", "content": "Hello! I'm Euler, your Socratic Tutor. Let's solve some problems together."}]
        st.session_state.user_answers = {q['id']: None for q in QUESTION_BANK}

    if 'problem_states' not in st.session_state:
        initialize_quiz()

    st.title("EULER: AI Socratic Tutor")
    st.write("---")

    # --- Main Layout ---
    main_col, profile_col = st.columns([2, 1])

    with main_col:
        st.subheader("Quiz Questions")
        for i, q in enumerate(QUESTION_BANK):
            with st.container(border=True):
                st.latex(q['latex'])
                st.session_state.user_answers[q['id']] = st.number_input("Your Answer:", key=f"ans_{q['id']}", step=None, value=st.session_state.user_answers.get(q['id']), format="%.3f")
                
                submit_col, hint_col = st.columns(2)
                with submit_col:
                    if st.button("Submit Answer", key=f"submit_{q['id']}"):
                        time_taken = time.time() - st.session_state.last_action_time
                        st.session_state.problem_states[q['id']]['attempts'] += 1
                        user_answer = st.session_state.user_answers.get(q['id'])
                        is_correct = 1 if user_answer is not None and abs(user_answer - q['answer']) < 0.001 else 0

                        current_action = { 'skill': q['skill'], 'problemType': q['problemType'], 'correct': is_correct, 'attemptCount': st.session_state.problem_states[q['id']]['attempts'], 'totalHints': st.session_state.problem_states[q['id']]['hints'], 'timeTaken_capped_95': min(time_taken, 113.0), 'prev_correct': st.session_state.prev_correct }
                        current_action['log_timeTaken'] = np.log(current_action['timeTaken_capped_95'] + 1)
                        current_action['time_per_attempt'] = current_action['timeTaken_capped_95'] / current_action["attemptCount"] if current_action["attemptCount"] > 0 else 0
                        
                        st.session_state.session_sequence.append(current_action)
                        st.session_state.prev_correct = is_correct
                        st.session_state.last_action_time = time.time()

                        sequence_df = pd.DataFrame(st.session_state.session_sequence)
                        predictions_df = predict_student_state(sequence_df, profiler_model, scalers, mappings)
                        
                        if not predictions_df.empty:
                            dominant_state = predictions_df['Dominant State'].iloc[-1]
                            with st.spinner("Tutor is thinking..."):
                                feedback = get_socratic_feedback(socratic_model, socratic_tokenizer, dominant_state, q['latex'], user_answer, is_correct)
                            
                            st.session_state.socratic_dialogue.append({"role": "user", "content": f"My answer to Q{i+1} was {user_answer}."})
                            st.session_state.socratic_dialogue.append({"role": "assistant", "content": feedback})
                        else:
                             st.warning("Learner Profiler could not make a prediction.")

                with hint_col:
                    if st.button("Get a Hint", key=f"hint_{q['id']}"):
                        st.session_state.problem_states[q['id']]['hints'] += 1
                        st.info(q['hint'])
                        st.session_state.socratic_dialogue.append({"role": "user", "content": f"I need a hint for Q{i+1}."})
                        st.session_state.socratic_dialogue.append({"role": "assistant", "content": f"Of course! Here is a hint: {q['hint']}"})


    with profile_col:
        st.subheader("Socratic Tutor Dialogue")
        dialogue_container = st.container(height=300, border=True)
        for message in st.session_state.socratic_dialogue:
            with dialogue_container.chat_message(name=message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        st.subheader("Learner Profile Analysis")
        if st.session_state.session_sequence:
            sequence_df = pd.DataFrame(st.session_state.session_sequence)
            predictions_df = predict_student_state(sequence_df, profiler_model, scalers, mappings)
            if not predictions_df.empty:
                display_df = pd.concat([sequence_df[['correct', 'attemptCount', 'totalHints']].reset_index(drop=True), predictions_df], axis=1)
                st.dataframe(display_df, use_container_width=True)
                latest_state = display_df['Dominant State'].iloc[-1]
                st.metric("Current Inferred State", latest_state)

    if st.sidebar.button("Restart Quiz"):
        for key in list(st.session_state.keys()):
            if key not in ['authentication_status', 'name', 'username']:
                 del st.session_state[key]
        st.rerun()

elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
elif st.session_state.get("authentication_status") is None:
    st.warning('Please enter your username and password')

