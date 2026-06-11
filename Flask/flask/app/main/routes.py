import os
import secrets
from flask import render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user, logout_user
from app import db
from app.main import main
from app.forms import EditProfileForm, DeleteAccountForm
from app.models import User


def save_profile_image(image_file):
    random_hex = secrets.token_hex(8)
    _, file_ext = os.path.splitext(image_file.filename)
    filename = random_hex + file_ext.lower()
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    image_file.save(file_path)
    return filename


@main.route('/')
def home():
    return render_template('home.html', title='Home')


@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', title='Profile', user=current_user)


@main.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(obj=current_user)
    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.bio = form.bio.data
        if form.profile_image.data:
            filename = save_profile_image(form.profile_image.data)
            current_user.profile_image = filename
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('main.profile'))
    return render_template('edit_profile.html', title='Edit Profile', form=form)


@main.route('/profile/delete', methods=['GET', 'POST'])
@login_required
def delete_account():
    form = DeleteAccountForm()
    if form.validate_on_submit():
        user_id = current_user.id
        logout_user()
        User.query.filter_by(id=user_id).delete()
        db.session.commit()
        flash('Your account has been deleted.', 'info')
        return redirect(url_for('main.home'))
    return render_template('confirm_delete.html', title='Delete Account', form=form)


@main.route('/api/me')
@login_required
def api_me():
    return jsonify({
        'id': current_user.id,
        'name': current_user.name,
        'email': current_user.email,
        'bio': current_user.bio,
        'profile_image': current_user.profile_image_url(),
    })


@main.route('/api/user/<int:user_id>')
@login_required
def api_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'bio': user.bio,
        'profile_image': user.profile_image_url(),
    })


@main.app_errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html', title='Page Not Found'), 404


@main.app_errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html', title='Server Error'), 500
