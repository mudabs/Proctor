"""Administration, user-management, and website blacklist routes."""

import json
import os
from datetime import datetime

from flask import flash, jsonify, redirect, render_template, request, session, url_for

from proctor.extensions import db
from proctor.models import (
    Blocked, Lecturers, Role, User, User_roles, assign_role_to_user, removed_role,
)
from . import admin

hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
name = ""
image_count = 0
capture_enabled = False


def block_websites():
    blocked_websites = [blocked.url for blocked in Blocked.query.all()]
    now = datetime.now()
    with open(hosts_path, "a") as hosts_file:
        hosts_file.write("\n\n")
        for website in blocked_websites:
            hosts_file.write("127.0.0.1 {}\n".format(website))
            hosts_file.write("127.0.0.1 www.{}\n".format(website))
    print("Websites blocked successfully at", now)


def unblock_websites():
    with open(hosts_path, "r") as hosts_file:
        lines = hosts_file.readlines()
    with open(hosts_path, "w") as hosts_file:
        for line in lines:
            if not any(website in line for website in [blocked.url for blocked in Blocked.query.all()]):
                hosts_file.write(line)
    print("Websites unblocked successfully")


@admin.route('/viewLecturers')
def viewLecturers():
    all_data = Lecturers.query.all()
    return render_template("manageLecturers.html", lecturers=all_data)

@admin.route('/display_images_admin')
def display_images_admin():
    all_data = User.query.all()

    return render_template('display_images_admin.html', Users = all_data)

@admin.route('/get_images', methods=['GET','POST'])
def get_images():
    name = request.form['user_name']
    images = []

    if name:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join("static", "images","known_images")
        images_dir = os.path.join(current_dir, static_dir)
        for filename in os.listdir(images_dir):
            if filename.startswith(name):
                images.append(filename)
    return json.dumps({"images": images})

@admin.route('/createLecturers', methods = ['POST'])
def createLecturers():
    if request.method == 'POST':
        lecturer = request.form['lecturerName']

        my_data = Lecturers(lecturer)
        db.session.add(my_data)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('viewLecturers'))

# update lecturers
@admin.route('/updateLecturers', methods = ['GET', 'POST'])
def updateLecturers():
    if request.method == 'POST':
        my_data = Lecturers.query.get(request.form.get('id'))

        my_data.name = request.form['name']
        db.session.commit()
        flash("Lecturer Data Updated Successfully")
    return redirect(url_for('viewLecturers'))


# Delete lecturers
@admin.route('/deleteLecturers/<id>/', methods=['GET', 'POST'])
def deleteLecturers(id):
    my_data = Lecturers.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewLecturers'))

# Lecturers---------------------------------------------------------------------------------------------------------------------------------------------

# Users---------------------------------------------------------------------------------------------------------------------------------------------

# User Roles
@admin.route('/users')
def user_list():
    users = User.query.all()
    return render_template('roles.html', users=users)

#insert data to mysql database via html forms
@admin.route('/viewUsers')
def viewUsers():
    # Questions data
    all_data = User.query.all()

    return render_template("manageUsers.html", Users = all_data)



#Create Users
@admin.route('/createUsers', methods = ['POST'])
def createUsers():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        userType = request.form['userType']
        password = generate_password_hash("default")
        imageStatus = "Unregistered"

        # Adding Users to DB
        my_data = User(name,email,password,userType,imageStatus)
        db.session.add(my_data)
        db.session.commit()

        flash("User Created Successfully")
        return redirect(url_for('display_images_admin'))

#update users
@admin.route('/updateUsers', methods = ['GET', 'POST'])
def updateUsers():
    if request.method == 'POST':
        my_data = User.query.get(request.form.get('id'))

        my_data.name = request.form['name']
        my_data.regNumber = request.form['regNumber']
        db.session.commit()
        flash("User Data Updated Successfully")
    return redirect(url_for('display_images_admin'))


# Delete users
@admin.route('/deleteUsers/<id>/', methods=['GET', 'POST'])
def deleteUsers(id):
    my_data = User.query.get(id)
    if my_data:  # Check if user exists before deleting
        name = my_data.name
        print("name",name)
        images = []
        if name:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            static_dir = os.path.join("static", "images","known_images")
            images_dir = os.path.join(current_dir, static_dir)

            for filename in os.listdir(images_dir):
                if filename.startswith(name):
                    newDir = os.path.join(images_dir, filename)
                    images.append(newDir)

            for image in images:
                os.remove(image)
                print("Removed file ",image)

        db.session.delete(my_data)
        db.session.commit()
        flash("User Deleted Successfully")
    else:
        flash("User not found!")
    return redirect(url_for('display_images_admin'))

# Users---------------------------------------------------------------------------------------------------------------------------------------------

# Course---------------------------------------------------------------------------------------------------------------------------------------------
#insert data to mysql database via html forms
# Roles and Login---------------------------------------------------------------------------------------------------------------------------------------


@admin.route('/roles', methods=['GET', 'POST'])
def manage_roles():
    if request.method == 'POST':
        role_name = request.form['name']  # Get role name from HTML form
        if role_name:
            role = Role(name=role_name)
            db.session.add(role)
            db.session.commit()
            flash(f'Role "{role.name}" created successfully!', 'success')
        else:
            flash('Please enter a role name!', 'error')
    roles = Role.query.all()
    users = User.query.all()
    return render_template('userroles.html', roles=roles,users=users)

@admin.route('/assign_roles/<int:user_id>', methods=['GET'])
def assign_roles(user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        flash('User not found!', 'error')
        return redirect(url_for('user_list'))  # Replace with your user list route

    assigned_roles = User_roles.query.filter_by(user_id=user_id)
    roles = Role.query.all()
    return render_template('assign_roles.html', user=user, roles=roles,assigned_roles=assigned_roles)

@admin.route('/assign_role/<int:user_id>', methods=['POST'])
def assign_role(user_id):
    user = User.query.get(user_id)
    role_id = request.form['role_id']
    role = Role.query.get(role_id)
    if not user or not role:
        flash('Invalid user or role!', 'error')
        return redirect(url_for('assign_roles', user_id=user_id))
    assign_role_to_user(user_id, role_id)
    flash(f'Role "{role.name}" assigned to user "{user.name}" successfully!', 'success')
    return redirect(url_for('assign_roles', user_id=user_id))

@admin.route('/remove_role/<int:user_id>/<int:role_id>', methods=['GET','POST'])
def remove_role(user_id, role_id):
    role = User_roles.query.filter_by(user_id=user_id,role_id=role_id)
    user = User.query.get(user_id)
    roles = Role.query.get(role_id)
    if not role:
        flash('Invalid user or role!', 'error')
        return redirect(url_for('assign_roles', user_id=user_id))
    removed_role(user_id, role_id)
    flash(f'Role removed from user  successfully!', 'success')
    return redirect(url_for('assign_roles', user_id=user_id))

# Roles and Login---------------------------------------------------------------------------------------------------------------------------------------
@admin.route('/edit_image')
def edit_image():
    user = User.query.filter_by(id=session['user_id']).one()
    return render_template('edit_image.html', user = user)

@admin.route('/deleteImages', methods=['GET','POST'])
def deleteImages():
    global name , image_count, capture_enabled
    if request.method == 'POST':
        if (request.form["del"] == "True"):
            images = []
            if name:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                static_dir = os.path.join("static", "images")
                images_dir = os.path.join(current_dir, static_dir)

                for filename in os.listdir(images_dir):
                    if filename.startswith(name):
                        newDir = os.path.join(images_dir, filename)
                        images.append(newDir)

                for image in images:
                    os.remove(image)
                    print("Removed file ",image)
    return redirect(url_for('captureImageReg'))

def black():
    blocked_websites = [
        "researchgate.net",
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "jstor.org",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "arxiv.org",
        "ssrn.com",
        "nature.com",
        "elsevier.com",
        "dl.acm.org",
        "scopus.com",
        "plos.org",
        "academic.oup.com",
        "tandfonline.com",
        "research.com",
        "researcher.com",
        "worldcat.org",
        "google.com",
        "bing.com",
        "yahoo.com",
        "duckduckgo.com",
        "baidu.com",
        "yandex.com",
        "ask.com",
        "ecosia.org",
        "startpage.com",
        "swisscows.com"
    ]

    # Add the URLs to the database
    for url in blocked_websites:
        blocked = Blocked(url=url)
        db.session.add(blocked)
    db.session.commit()
    # Block websites after adding them to the database
    unblock_websites()
    block_websites()
    flash('Websites blocked successfully', 'success')
    return redirect(url_for('blacklist'))
    


# Route to unblock all websites
def unblock():
    unblock_websites()
    # Delete all records from the Blocked table
    Blocked.query.delete()
    db.session.commit()
    # Unblock websites after removing them from the database
    
    flash('Websites unblocked successfully', 'success')
    return redirect(url_for('blacklist'))
