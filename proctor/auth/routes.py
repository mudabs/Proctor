"""Authentication and user image routes."""

import base64
import os

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from werkzeug.security import check_password_hash, generate_password_hash

from .. import state
from ..extensions import db
from ..models import User
from . import auth


def check_names(username):
    """Return whether a user name already exists."""
    return User.query.filter_by(name=username).first() is not None


@auth.route('/resetPassword', methods=['POST'])
def resetPassword():
    user = User.query.filter_by(id=session['user_id']).one()
    if request.method == 'POST' and request.form["p1"] == request.form["p2"]:
        user.password = generate_password_hash(request.form["p1"])
        try:
            db.session.commit()
        except Exception as e:
            print("Error committing changes:", e)
            db.session.rollback()

    return redirect(url_for('edit_profile'))

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        userType = request.form['userType']
        try:
            user = User.query.filter_by(email=email).one()
            stored_password = user.password or ""
            if stored_password.startswith(("scrypt:", "pbkdf2:")):
                password_matches = check_password_hash(stored_password, password)
            else:
                # Migrate legacy plaintext credentials after one successful login.
                password_matches = stored_password == password

            if password_matches and user.userType == userType and user.email == email:
                session['user_id'] = user.id
                session['username'] = user.name
                state.me = session['username']
                if not stored_password.startswith(("scrypt:", "pbkdf2:")):
                    user.password = generate_password_hash(password)
                    db.session.commit()
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid credentials!', 'danger')
                return render_template('login.html', error="Incorrect password or User Type.")
        except NoResultFound:
            flash('User does not exist.', 'danger')
            return render_template('login.html', error="User does not exist.")
    return render_template('login.html', error=None)
# Login------------------------------------------------------------------------------------------------------------------------------------------------

# Logout-----------------------------------------------------------------------------------------------------------------------------------------------
@auth.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logout successful!', 'success')
    return redirect(url_for('auth.login'))
# Logout-----------------------------------------------------------------------------------------------------------------------------------------------


#  Register---------------------------------------------------------------------------------------------------------------------------------------------

@auth.route('/register', methods=['GET', 'POST'])
def register():
    # global capture_enabled, name, id, image_count
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        userType = request.form['userType']
        imageStatus = "Registered"
        try:
            new_user = User(name=name, email=email, password=generate_password_hash(password), userType=userType,imageStatus=imageStatus)
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id
            return redirect(url_for('home'))
        except IntegrityError:
            db.session.rollback()
            return render_template('register.html', error="User already exists.")
    return render_template('register.html', error=None)

@auth.route('/captureImage', methods=['GET','POST'])
def captureImage():
    name = secure_filename(request.form.get('name', '').strip())
    data_url = request.form.get('imageDataURL', '')
    if not name or ',' not in data_url:
        return jsonify({'message': 'A name and captured image are required'}), 400
    try:
        image_data = base64.b64decode(data_url.split(',', 1)[1], validate=True)
    except (ValueError, TypeError):
        return jsonify({'message': 'The captured image is invalid'}), 400
    if len(image_data) > current_app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'message': 'The captured image is too large'}), 413
    image_dir = current_app.config['PROCTOR_DATA_DIR'] / 'known_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f'{name}.jpg'
    image_path.write_bytes(image_data)
    return jsonify({'message': 'User image captured successfully'})

@auth.route('/recaptureImage', methods=['GET','POST'])
def recaptureImage():
    formData = request.form
    message = ''
    print("imageData",formData)
    if 'imageDataURL' in formData:
        imageData = formData['imageDataURL'].split(',')[1]

        user = User.query.filter_by(id=session['user_id']).one()
        imageName = user.name+'.jpg'

        with open(os.path.join('static/images/known_images', imageName), 'wb') as f:
            f.write(base64.decodebytes(imageData.encode()))
            message = 'User image captured successfully'

            print(user.imageStatus)
            if (user.imageStatus == "Unregistered"):
                user.imageStatus = "Registered"
                db.session.commit()

        return jsonify({'message': message})
    return redirect(url_for('edit_image'))

#  Register---------------------------------------------------------------------------------------------------------------------------------------------
