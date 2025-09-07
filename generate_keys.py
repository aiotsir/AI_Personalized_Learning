# his is a helper script that you will run only once to securely hash the passwords for your users and update the config.yaml file.

# import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher


# --- This script is for generating hashed passwords ---
# --- Run this script once to set up your user passwords ---

# You can add as many passwords as you have users
passwords_to_hash = ['hi@123'] #, 'def'] 

# hashed_passwords = stauth.Hasher(passwords_to_hash).generate()
# CORRECTED: The Hasher class is initialized first, then generate is called with the passwords.
# hashed_passwords = stauth.Hasher().generate(passwords_to_hash)
hashed_passwords = Hasher(passwords_to_hash).generate()

print("Copy the following into your config.yaml file under the 'password' field for each user:")
print(hashed_passwords)

# Example Output will look like:
# ['$2b$12$....', '$2b$12$....']
# Copy the first hashed password for the first user, the second for the second user, and so on.