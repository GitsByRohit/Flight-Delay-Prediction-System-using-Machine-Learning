from werkzeug.security import generate_password_hash, check_password_hash
from app.database import get_db_connection


# ==============================
# REGISTER USER
# ==============================
def register_user(username, email, password):

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if email already exists
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return False, "Email already registered"

    hashed_password = generate_password_hash(password)

    cursor.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, hashed_password)
    )

    conn.commit()
    conn.close()

    return True, "User registered successfully"


# ==============================
# LOGIN USER
# ==============================
def login_user(email, password):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()

    conn.close()

    if user and check_password_hash(user["password"], password):
        return True, user

    return False, "Invalid email or password"
