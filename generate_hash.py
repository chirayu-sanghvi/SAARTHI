import hashlib
def generate_hashed_password(password, salt=None):
    if salt is None:
        salt = hashlib.sha256().hexdigest()[:16]
    
    salted_password = password + salt
    hashedd_password = hashlib.sha256(salted_password.encode()).hexdigest()
    return hashedd_password, salt

password="testone"
hashed_password, salt = generate_hashed_password(password)
print(hashed_password)
print(salt)