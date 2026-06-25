from flask import Blueprint, render_template, request, redirect, url_for, session, flash

from app.services.auth_service import register_user, login_user

auth_bp = Blueprint("auth", __name__)


# ==============================
# SIGNUP
# ==============================
@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        success, message = register_user(username, email, password)

        if success:
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("auth.login"))
        else:
            flash(message, "danger")

    return render_template("signup.html")


# ==============================
# LOGIN
# ==============================
@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        success, result = login_user(email, password)

        if success:
            session["user_id"] = result["id"]
            session["username"] = result["username"]

            flash("Login successful!", "success")
            return redirect(url_for("prediction.dashboard"))

        else:
            flash(result, "danger")

    return render_template("login.html")


# ==============================
# LOGOUT
# ==============================
@auth_bp.route("/logout")
def logout():

    session.clear()
    flash("Logged out successfully.", "info")

    return redirect(url_for("auth.login"))
