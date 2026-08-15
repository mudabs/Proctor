"""Course, question, exam, quiz, and result routes."""

import random
import string
import threading
from datetime import datetime

from flask import flash, jsonify, redirect, render_template, request, session, url_for

from app import (
    app,
    configure_app_and_access_session,
    detectSound,
    drawGraph,
    drawSoundGraph,
)
from proctor.admin.routes import black, unblock
from proctor import state
from proctor.extensions import db
from proctor.models import (
    Answers, CorrectAnswers, Course, Enrollment, Exam, Lecturers, Marks,
    ProctorSession, Questions, Quiz, QuizCompletion, QuizQuestions, User,
)
from . import courses

average_threshold = None
# Course---------------------------------------------------------------------------------------------------------------------------------------------
#insert data to mysql database via html forms
@courses.route('/viewCourses')
def viewCourses():
    # Courses data
    all_data = Course.query.all()
    lecturers = Lecturers.query.all()
    return render_template("manageCourses.html", courses = all_data, lecturers = lecturers, user =  session['username'])


@courses.route('/courseAdminSide/<courseId>', methods=['GET'])
def courseAdminSide(courseId):
    courses = Course.query.filter_by(id = courseId).all()
    courseTitle=''
    for c in courses:
        courseTitle = c.courseTitle
    enrollment = Enrollment.query.filter_by(course_id = courseId).all()
    users = User.query.all()
    quiz = Quiz.query.filter_by(courseId = courseId).all()
    return render_template("courseAdminSide.html",courses = courses,enrollment=enrollment,users=users,quiz=quiz,courseTitle=courseTitle)


@courses.route('/courseUsersSide/<courseId>', methods=['GET'])
def courseUsersSide(courseId):
    qq = []
    courses = Course.query.filter_by(id = courseId).all()
    enrollment = Enrollment.query.filter_by(course_id = courseId).all()
    users = User.query.all()
    quizes = Quiz.query.filter_by(courseId = courseId).all()
    marks = Marks.query.filter_by(userId=session['user_id'])
    for q in quizes:
        for mark in marks:
            if q.id == mark.quizId:
                pass
            else:
                qq.append(q)

    return render_template("courseUserSide.html",courses = courses,enrollment=enrollment,users=users,quizes=quizes,allquizes = quizes,marks=marks, userId =  session['user_id'])

#insert data to mysql database via html forms
@courses.route('/viewCoursesUser')
def viewCoursesUser():
    # Courses data
    all_data = Course.query.filter_by(lecturerId=session['user_id']).all()
    lecturers = Lecturers.query.all()

    return render_template("manageCoursesUser.html", courses = all_data, lecturers = lecturers, user =  session['username'])


def generateCourseCode():
    """Generates a 6-character word with random letters and numbers."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(6))


#insert data to mysql database via html forms
@courses.route('/createCourse', methods = ['GET','POST'])
def createCourse():
    if request.method == 'POST':
        course = request.form['courseName']
        courseCode = generateCourseCode()
        # Adding Course to DB
        my_data = Course(course,session['user_id'],courseCode)
        db.session.add(my_data)
        db.session.commit()

        flash("Course Created Successfully")
    return redirect(url_for('courseAdminSide',courseId = my_data.id))


#insert data to mysql database via html forms
@courses.route('/joinCourse', methods = ['POST'])
def joinCourse():
    courseId=''
    if request.method == 'POST':
        courseCode = request.form['courseCode']
        # Adding Course to DB
        courses = Course.query.all()
        for course in courses:
            if courseCode == course.courseCode:
                courseId=course.id
                my_data = Enrollment(session['user_id'],course.id)
                db.session.add(my_data)
                db.session.commit()
                message = "You have enrolled for couse Successfully"
            else:
                message = "Course Not Found"

    flash(message)
    return redirect(url_for('courseUsersSide', courseId=courseId))


#insert data to mysql database via html forms
@courses.route('/enroll', methods = ['POST'])
def enroll():
    if request.method == 'POST':
        course = request.form['course']
        student = request.form['student']
        # Adding Course to DB
        my_data = Enrollment(student,course)
        db.session.add(my_data)
        db.session.commit()

        flash("Lecturer Added to Course Successfully")
    return redirect(url_for('courseUsersSide', courseId=course))

# Unenroll Course
@courses.route('/unenroll/<user_id>/<course_id>', methods=['GET', 'POST'])
def unenroll(user_id,course_id):
    my_data = Enrollment.query.get((user_id,course_id))
    # my_data = Enrollment.query.filter_by(user_id=user_id, course_id=course_id).first()

    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Unenrolled Successfully")
    else:
        flash("Error! not found!")
    return redirect(url_for('home'))

#update Course
@courses.route('/updateCourse', methods = ['GET', 'POST'])
def updateCourse():
    if request.method == 'POST':
        my_data = Course.query.get(request.form.get('id'))

        my_data.courseTitle = request.form['courseName']
        my_data.lecturerId = request.form['lecturerId']
        db.session.commit()
        flash("Course Data Updated Successfully")
    return redirect(url_for('viewCourses'))


# Delete Course
@courses.route('/deleteCourse/<id>/', methods=['GET', 'POST'])
def deleteCourse(id):
    my_data = Course.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Course Deleted Successfully")
    else:
        flash("Course not found!")
    return redirect(url_for('home'))
# Course---------------------------------------------------------------------------------------------------------------------------------------------



# Exam---------------------------------------------------------------------------------------------------------------------------------------------
#insert data to mysql database via html forms
@courses.route('/manageResults/<quizId>')
def manageResults(quizId):
    quiz = Quiz.query.filter_by(id=quizId)
    courseId=''
    courseTitle = ''
    quizTitle = ''
    for q in quiz:
        courseId = q.courseId
        quizTitle = q.topic
    course = Course.query.filter_by(id = courseId)
    for c in course:
        courseTitle = c.courseTitle

    # users = User.query.all()
    # marks = Marks.query.filter_by(quizId = quizId)
    users = User.query.join(Marks, User.id == Marks.userId).filter(Marks.quizId == quizId).all()
    # users = User.query.join(Marks, User.id == Marks.userId).outerjoin(ProctorSession, ProctorSession.userId == User.id).filter(Marks.quizId == quizId).all()
    marks = Marks.query.join(User, Marks.userId == User.id).filter(Marks.quizId == quizId).all()
    cheatthreshold = ProctorSession.query.all()


   

    return render_template("manageResults.html",courseTitle=courseTitle,quizTitle=quizTitle,users=users,cheatthreshold=cheatthreshold)

@courses.route('/fetch_data', methods=['GET'])
def fetch_data():
    topic = request.args.get('topic')
    my_data = Quiz.query.filter_by(topic = topic).all()
    data=[]
    # ... query database based on selected topic ...
    if topic == "0":
        data.append({
        'points': " ",
        'dueDate': " ",
        'instructions': " ",
        'proctor': " "
        })
    else:
        for quiz in my_data:
            data.append({
                'points': quiz.totalPoints,
                'dueDate': quiz.date,
                'instructions': quiz.instructions,
                'proctor': quiz.proctor
            })
    return jsonify(data)

#insert data to mysql database via html forms
@courses.route('/createExams', methods = ['POST'])
def createExams():
    if request.method == 'POST':
        code = request.form['examName']
        courseId = request.form['courseId']
        date = request.form['date']
        durationHours = request.form['durationHours']
        durationMinutes = request.form['durationMinutes']
        duration = float(durationHours) + (float(durationMinutes)/60)
        # Adding Exam to DB
        my_data = Exam(code,courseId,date,duration)
        db.session.add(my_data)
        db.session.commit()

        flash("Exam Created Successfully")
    return redirect(url_for('viewExams'))


#insert data to mysql database via html forms
@courses.route('/createQuiz', methods = ['POST'])
def createQuiz():
    if request.method == 'POST':
        courseId = request.form['course']
        topic = request.form['topic']
        totalPoints = request.form['totalPoints']
        date = request.form['duedate']
        durationHours = request.form['durationHours']
        durationMinutes = request.form['durationMinutes']
        duration = float(durationHours) + (float(durationMinutes)/60)
        instructions = request.form['instructions']
        if request.form['proctor']:
            proctor = "True"
        else:
            proctor = "False"

        # Adding Quiz to DB
        my_data = Quiz(courseId,topic,totalPoints,date,duration,instructions,proctor)
        db.session.add(my_data)
        db.session.commit()

        flash("Exam Created Successfully")
    return redirect(url_for('addQuestions',courseId=courseId,topic=topic,totalPoints=totalPoints,date=date,duration=duration,instructions=instructions,proctor=proctor))


#update Exam
@courses.route('/updateExams', methods = ['GET', 'POST'])
def updateExams():
    if request.method == 'POST':
        my_data = Exam.query.get(request.form.get('id'))

        my_data.courseTitle = request.form['courseName']
        my_data.lecturerId = request.form['lecturerId']
        db.session.commit()
        flash("Exam Data Updated Successfully")
    return redirect(url_for('viewExams'))


# Delete Exam
@courses.route('/deleteExams/<id>/', methods=['GET', 'POST'])
def deleteExams(id):
    my_data = Exam.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewExams'))

@courses.route('/setTimer/<quizId>')
def setTimer(quizId):
    configure_app_and_access_session(app, session)
    black()
    return redirect(url_for('takeQuiz',quizId=quizId))

@courses.route('/takeQuiz/<quizId>')
def takeQuiz(quizId):
    quiz = Quiz.query.filter_by(id = quizId).all()
    questionLink = QuizQuestions.query.filter_by(quizId = quizId).all()
    questions = Questions.query.filter_by(quizId=quizId).all()

    answers = Answers.query.all()
    session['quiz'] = quiz[0].topic
    session['duration'] = quiz[0].duration
    session['myDuration'] = datetime.now().timestamp()

    user_timezone = session['expiration_time'].tzinfo
    numQuestions = len(questions)

    now_with_timezone = datetime.now(user_timezone)

    # Reset shared proctoring state for each quiz attempt.
    state.stop_detection = False
    state.cheating_scores.clear()

    # Start sound detection in a separate thread
    sound_thread = threading.Thread(target=detectSound)
    sound_thread.start()


    if now_with_timezone > session['expiration_time']:
        # Handle quiz expiration (e.g., redirect to a different page, display a message)
        state.stop_detection = True
        drawGraph()
        drawSoundGraph()
        return redirect(url_for('home'))  # Example redirect
    
    return render_template("takeQuiz.html",quiz=quiz,quizId=quizId,questionLink=questionLink,questions=questions,answers=answers,numQuestions=numQuestions,userId = session['user_id'])

average_threshold = None
# Submit Exam
@courses.route('/quizCompletion', methods=['POST'])
def quizCompletion():
    global average_threshold
    state.stop_detection = True
    quizName =''
    courseName =''
    if request.method == 'POST':
        unblock()

        attemptedAnswers=[]
        status="0"
        totalmarks = 0

        quizId = request.form["quizId"]
        userId = request.form["userId"]

        # Marks
        mark=0
        correctAnswers = CorrectAnswers.query.filter_by(quizId=quizId).all()
        questions = Questions.query.filter_by(quizId=quizId).all()
        quizes = Quiz.query.filter_by(id=quizId).all()
        for q in quizes:
            course = Course.query.filter_by(id = q.courseId).all()
            for c in course:
                if c.id == q.courseId:
                    quizName = q.topic
                    courseName = c.courseTitle


        for question in questions:
            totalmarks = totalmarks + float(question.points)
            if question.answerType == "Multiple Choice":
                mcq = "mcq" + str(question.id)
                try:
                    answer=request.form[mcq]

                    for correctAnswer in correctAnswers:
                        if correctAnswer.questionId == question.id:

                            if answer == correctAnswer.answer:
                                status = str(question.points)
                            else:
                                status = "0"
                            mark += float(status)

                    attemptedAnswers.append(answer)
                except Exception as e:
                    print(attemptedAnswers)
            else:
                status="pending"

            now = datetime.now()
            myduration = now.timestamp() - session['myDuration']
            session['myDuration'] = int(myduration)
            my_data = QuizCompletion(quizId=quizId,userId=userId,questionId = question.id,answer=answer, status = status)
            db.session.add(my_data)
            db.session.commit()
        marks = Marks(quizId = quizId,userId = userId,mark = mark, duration = session['myDuration'], totalmark = totalmarks)
        db.session.add(marks)
        db.session.commit()

        drawGraph()
        drawSoundGraph()
        # Calculate average cheating threshold (if any scores exist)
        
        if state.cheating_scores:
            average_threshold = sum(state.cheating_scores) / len(state.cheating_scores)
            if (average_threshold < 0.6):
                average_threshold = average_threshold - 0.3
            print(f"Average cheating threshold: {average_threshold}")
            now = datetime.now()

            my_data = ProctorSession(session['user_id'],average_threshold,now)
            db.session.add(my_data)
            db.session.commit()

        

    return redirect(url_for('userResults',quizId=quizId))

@courses.route('/displayGraph/<userId>', methods=['POST'])
def displayGraph(userId):

    # if request.method == 'POST':

    return render_template('userResults.html')



@courses.route('/userResults/<quizId>', methods=['GET'])
def userResults(quizId):

    quiz = Quiz.query.filter_by(id = quizId).all()
    questionLink = QuizQuestions.query.filter_by(quizId = quizId).all()
    questions = Questions.query.filter_by(quizId=quizId).all()
    answers = Answers.query.all()
    session['quiz'] = quiz[0].topic
    session['duration'] = quiz[0].duration
    numQuestions = len(questions)
    quizCompletion = QuizCompletion.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()


    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    return render_template('userResults.html', quiz=quiz,quizId=quizId,questionLink=questionLink,questions=questions,answers=answers,numQuestions=numQuestions,userId = session['user_id'],quizCompletion=quizCompletion,marks=marks)




# Exam---------------------------------------------------------------------------------------------------------------------------------------------





# Login------------------------------------------------------------------------------------------------------------------------------------------------

@courses.route('/viewAllQuestions/<int:quizId>', methods=['GET', 'POST'])
def viewAllQuestions(quizId):

    return render_template("manageResults.html")



@courses.route('/addQuestions/<courseId>',methods=['GET','POST'])
def addQuestions(courseId):
    lecturerId = session['user_id']
    num_inputs = request.args.get('num_inputs', 1, type=int)
    course = Course.query.filter_by(id=courseId).all()
    quiz = Quiz.query.filter_by(courseId = courseId).all()
    quizI = Quiz.query.filter_by(courseId = courseId).first()
    quizId=quizI
    questions = Questions.query.filter_by(quizId=quizId.id)
    answers = Answers.query.all()


    return render_template("addQuestions.html", quiz=quiz, num_inputs=num_inputs, lecturerId=lecturerId,courseId=courseId,course=course, questions=questions,answers=answers)

@courses.route('/addQuestion', methods=['POST'])
def addQuestion():
    correctAnswer = ""
    if request.method == 'POST':
        question = request.form['question']
        courseId = request.form['courseId']
        userId = session['user_id']
        answerType = request.form['answerType']
        topic = request.form['topic']
        points = request.form['qpoints']
        quiz = Quiz.query.filter_by(topic=topic, courseId=courseId).first()
        if quiz is not None:
            quizId = quiz.id
        else:
            # Handle the case where no quiz is found (e.g., display an error message)
            pass

        my_data = Questions(question=question,courseId=courseId,userId=userId,answerType=answerType,topic=topic,points=points,quizId=quizId)
        db.session.add(my_data)
        db.session.commit()
        quizAllocation = QuizQuestions(quizId=quizId,questionId=my_data.id)
        db.session.add(quizAllocation)
        db.session.commit()
        answers = []
        print(my_data.id)
        if request.form["answerType"] == "Multiple Choice":

            for i in range(1, int(request.form['num_inputs']) + 1):
                ans = request.form[f'answer{i}']
                if ans:  # Check if input field is not empty
                    answer = Answers(answer=ans, questionId =  my_data.id)
                    answers.append(answer)
            db.session.add_all(answers)
            db.session.commit()


            answerPlaceHolder = request.form["checkAnswer"]
            correctAnswer = request.form[answerPlaceHolder]
            print(correctAnswer)
            answerVar = CorrectAnswers(answer = correctAnswer,questionId=my_data.id,quizId=quizId)
            db.session.add(answerVar)
            db.session.commit()


    return redirect(url_for('addQuestions',courseId=courseId))

#Create Questions
@courses.route('/createQuestions', methods = ['POST'])
def createQuestions():
    if request.method == 'POST':
        question = request.form['question']
        course = request.form['courseId']
        lecturer = request.form['lecturerId']
        answer1 = request.form['answer1']
        answer2 = request.form['answer2']
        answer3 = request.form['answer3']
        answer4 = request.form['answer4']

        # Adding Question to DB
        my_data = Questions(question, course, lecturer)
        db.session.add(my_data)
        db.session.commit()

        # Adding ANswers to DB
        answers = [answer1, answer2, answer3, answer4]
        for answer_text in answers:
            my_answer = Answers(answer_text, questionId=my_data.id)  # Use my_data.id after commit
            db.session.add(my_answer)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('addQuestions',course_id=course))

#Update questions
@courses.route('/updateQuestions', methods = ['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Questions.query.get(request.form.get('id'))
        my_data2 = Answers.query.filter_by(questionId=request.form.get('id')).all()
        course = request.form['courseId']

        my_data.question = request.form['question']
        for answer in my_data2:
            # Update individual answer based on form data
            answer.answer = request.form.get(f"answer_{answer.id}")  # Access answer-specific form field

        db.session.commit()
        flash("Question Updated Successfully")
    return redirect(url_for('viewQuestions',course_id=course))


#Delete questions
@courses.route('/deleteQuestions/<id>/<course_id>', methods = ['GET', 'POST'])
def deleteQuestions(id,course_id):
    my_data = Questions.query.get(id)
    if my_data:  # Check if questions exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Questions Deleted Successfully")
    else:
        flash("Questions not found!")
    return redirect(url_for('addQuestions', courseId = course_id))
# Questions---------------------------------------------------------------------------------------------------------------------------------------------
