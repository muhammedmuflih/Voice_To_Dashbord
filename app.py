from flask import Flask, render_template, redirect, url_for, flash, session, request, send_from_directory
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import subprocess
import os
import datetime
import random
import threading
import time

# --- Flask App Initialization and Configuration ---
app = Flask(__name__)

# Replace with a strong, random key (use environment variables in production)
app.config['SECRET_KEY'] = 'fceb2fb6363760a0aa1dd168262c3f10b587cf425edbe23bbfbe07c50897ee25'

# Flask-SQLAlchemy Configuration for SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'muflih20298@gmail.com'  # <-- REPLACE
app.config['MAIL_PASSWORD'] = 'jxlslrrumzrjoitt'       # <-- REPLACE

db = SQLAlchemy(app)
mail = Mail(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Database Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.email}', '{self.id}')"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Form Classes ---
class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already registered. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class OTPForm(FlaskForm):
    otp = StringField('OTP', validators=[DataRequired(), Length(min=6, max=6)])
    submit = SubmitField('Verify OTP')

# --- Helper Function for Sending OTP Email ---
def send_otp_email(user_email, otp):
    """Sends a one-time password to the user's email."""
    try:
        msg = Message(
            subject='Your Registration OTP',
            sender=app.config['MAIL_USERNAME'],
            recipients=[user_email]
        )
        msg.body = f"Your one-time password (OTP) is: {otp}"
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# --- Function to run Streamlit in a separate thread ---
def run_streamlit():
    time.sleep(2)  # Give Flask time to start
    st_path = "streamlit" if os.name == 'nt' else "streamlit"
    subprocess.Popen([st_path, "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"])

# --- Routes ---
@app.route("/", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.', 'danger')

    return render_template('login.html', form=form)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Store user info temporarily in session
        session['registration_data'] = {
            'email': form.email.data,
            'password': form.password.data,
        }
        
        # Generate OTP and expiry (5 minutes)
        otp = str(random.randint(100000, 999999))
        expiry_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        
        session['otp'] = otp
        session['otp_expiry'] = expiry_time.isoformat()

        if send_otp_email(form.email.data, otp):
            flash('A one-time password has been sent to your email. Please verify within 5 minutes.', 'info')
            return redirect(url_for('verify_registration_otp'))
        else:
            flash('Failed to send verification email. Please try again later.', 'danger')
    
    return render_template('register.html', form=form)

@app.route("/verify_registration_otp", methods=['GET', 'POST'])
def verify_registration_otp():
    if 'registration_data' not in session or 'otp' not in session or 'otp_expiry' not in session:
        flash('Invalid request. Please start the registration process again.', 'warning')
        return redirect(url_for('register'))

    form = OTPForm()
    if form.validate_on_submit():
        stored_otp = session.get('otp')
        stored_expiry_str = session.get('otp_expiry')

        # Convert expiry string to datetime
        stored_expiry = datetime.datetime.fromisoformat(stored_expiry_str)

        # Check expiry
        if datetime.datetime.utcnow() > stored_expiry:
            session.pop('otp', None)
            session.pop('otp_expiry', None)
            flash('OTP has expired. Please register again.', 'danger')
            return redirect(url_for('register'))

        if form.otp.data == stored_otp:
            user_data = session.pop('registration_data')
            session.pop('otp', None)
            session.pop('otp_expiry', None)

            hashed_password = bcrypt.generate_password_hash(user_data['password']).decode('utf-8')
            user = User(email=user_data['email'], password=hashed_password)
            db.session.add(user)
            db.session.commit()

            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP. Please try again.', 'danger')
            return redirect(url_for('verify_registration_otp'))

    return render_template('verify_otp.html', form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    # Start Streamlit in a separate thread if not already running
    if not hasattr(app, 'streamlit_started'):
        app.streamlit_started = True
        thread = threading.Thread(target=run_streamlit)
        thread.daemon = True
        thread.start()
    
    # Redirect to the Streamlit dashboard
    return redirect("http://localhost:8501")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)