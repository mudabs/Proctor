"""SQLAlchemy models for Proctor."""

from sqlalchemy import orm

from .extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    userType = db.Column(db.String(20))
    imageStatus = db.Column(db.String(20))

    def __init__(self, name, email, password,userType,imageStatus):
        self.name = name
        self.email = email
        self.password = password
        self.userType = userType
        self.imageStatus = imageStatus

    #  Define the relationship with Marks class (assuming a one-to-many relationship)
    marks = orm.relationship("Marks", backref="user")

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)

class Questions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500))
    answerId = db.Column(db.Integer, db.ForeignKey('answers.id'))
    userId = db.Column(db.Integer, db.ForeignKey('user.id'))
    answerType = db.Column(db.String(20))
    topic = db.Column(db.String(100))
    points = db.Column(db.Integer)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id'))

    def __init__(self,question, courseId, userId, answerType, topic,points,quizId):  # Accept both arguments
        self.question = question
        self.courseId = courseId
        self.userId = userId
        self.answerType = answerType
        self.topic = topic
        self.points = points
        self.quizId = quizId

    # def __init2__(self, answerId):  # Accept both arguments
    #     self.answerId = answerId

    def __repr__(self):
        return f"<Question {self.quizId}>"

class Answers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(255))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))


    def __init__(self, answer, questionId):  # Accept both arguments
        self.answer = answer
        self.questionId = questionId

    def __repr__(self):
        return f"<Answer {self.answer}>"


class CorrectAnswers(db.Model):
    __tablename__ = 'correctanswers'
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(255))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id', ondelete='CASCADE'))


    def __init__(self, answer, questionId, quizId):  # Accept both arguments
        self.answer = answer
        self.questionId = questionId
        self.quizId = quizId

    def __repr__(self):
        return f"<Answer {self.answer}>"

class ProctorSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('user.id'))
    percentage = db.Column(db.String(255))
    time = db.Column(db.DateTime(255))

    def __init__(self, userId, percentage, time):  # Accept both arguments
        self.userId = userId
        self.percentage = percentage
        self.time = time

    def __repr__(self):
        return f"<Answer {self.percentage}>"

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    courseTitle = db.Column(db.String(255))
    lecturerId = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="SET NULL"))
    # examId = db.Column(db.Integer, db.ForeignKey('exam.id', ondelete="CASCADE"))
    courseCode = db.Column(db.String(10), unique=True)

    def __init__(self, courseTitle, lecturerId, courseCode):  # Accept both arguments
        self.courseTitle = courseTitle
        self.lecturerId = lecturerId
        self.courseCode = courseCode

    def __repr__(self):
        return f"<Answer {self.courseTitle}>"

class Enrollment(db.Model):
    __tablename__ = 'enrollments'  # Custom table name for clarity
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), primary_key=True)

    # Additional columns for enrollment specific data (optional)
    # enrolment_date = Column(DateTime, default=datetime.utcnow)
    # status = Column(String(20))  # enrolled, completed, etc.

    def __init__(self, user_id, course_id):
        self.user_id = user_id
        self.course_id = course_id

    def __repr__(self):
        return f"<Enrollment user_id: {self.user_id}, course_id: {self.course_id}>"

class Lecturers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))

    def __init__(self, name):  # Accept both arguments
        self.name = name

    def __repr__(self):
        return f"<Answer {self.name}>"

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(255))
    courseId = db.Column(db.Integer, db.ForeignKey('course.id',ondelete="CASCADE"))
    date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float, nullable=False)



    def __init__(self, code,courseId,date,duration):  # Accept both arguments
        self.code = code
        self.courseId = courseId
        self.date=date
        self.duration=duration

    def __repr__(self):
        return f"<Answer {self.answer_text}>"

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    courseId = db.Column(db.Integer, db.ForeignKey('course.id',ondelete="CASCADE"))
    topic = db.Column(db.String(100), nullable=False)
    totalPoints = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    instructions = db.Column(db.String(200), nullable=False)
    proctor=db.Column(db.String(20), nullable=False)

    def __init__(self, courseId,topic,totalPoints,date,duration,instructions,proctor):  # Accept both arguments
        self.courseId = courseId
        self.topic=topic
        self.totalPoints=totalPoints
        self.date=date
        self.duration=duration
        self.instructions = instructions
        self.proctor=proctor

    def __repr__(self):
        return f"<Answer {self.id}>"

class QuizQuestions(db.Model):
    __tablename__ = 'quizquestions'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id',ondelete="CASCADE"))

    def __init__(self, quizId,questionId):  # Accept both arguments
        self.quizId = quizId
        self.questionId=questionId

    def __repr__(self):
        return f"<Quiz Id: {self.quizId}, Question Id: {self.questionId}>"

class Marks(db.Model):
    __tablename__ = 'marks'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    mark = db.Column(db.Float)
    duration = db.Column(db.Integer)
    totalmark = db.Column(db.Float)

    def __init__(self, quizId,userId,mark,duration,totalmark):  # Accept both arguments
        self.quizId = quizId
        self.userId=userId
        self.mark = mark
        self.duration = duration
        self.totalmark = totalmark

    def __repr__(self):
        return f"<Quiz Id: {self.quizId}"

class QuizCompletion(db.Model):
    __tablename__ = 'quizcompletion'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id',ondelete="CASCADE"))
    answer = db.Column(db.String(500), nullable=False)
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    status = db.Column(db.String(20), nullable=False)

    def __init__(self, quizId,userId,questionId,answer,status):  # Accept both arguments
        self.quizId = quizId
        self.questionId=questionId
        self.answer = answer
        self.userId=userId
        self.status=status

    def __repr__(self):
        return f"<Completion Id: {self.id}>"

class UserCompletion(db.Model):
    __tablename__ = 'usercompletion'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    mark = db.Column(db.Float, nullable=False)
    quizStatus = db.Column(db.String(20), nullable=False)

    def __init__(self, quizId,userId,mark,quizStatus):  # Accept both arguments
        self.quizId = quizId
        self.userId=userId
        self.mark=mark
        self.quizStatus=quizStatus

    def __repr__(self):
        return f"<Completion Id: {self.id}>"

# Define a many-to-many relationship between User and Role
class User_roles(db.Model):
    user_id=db.Column( db.Integer, db.ForeignKey('user.id'))
    role_id=db.Column( db.Integer, db.ForeignKey('role.id'))
    __table_args__ = (db.PrimaryKeyConstraint('user_id', 'role_id'),)

    def __init__(self, user_id,role_id):  # Accept both arguments
        self.user_id = user_id
        self.role_id = role_id

    def __repr__(self):
        return f"<Answer {self.user_id}>"

def assign_role_to_user(userId, roleId):
    user_data = User_roles.query.filter_by(user_id=userId,role_id=roleId).first()
    if user_data:
        print("Error")
        return
    else:
        my_data = User_roles(userId, roleId)
        db.session.add(my_data)
        db.session.commit()
        return

def removed_role(userId, roleId):
    user_data = User_roles.query.filter_by(user_id=userId,role_id=roleId).first()
    if not user_data:
        print("Error")
        return
    else:
        db.session.delete(user_data)
        db.session.commit()
        return

def has_role(self, role):
    return role in self.roles

User.has_role = has_role

# Define the Blocked model
class Blocked(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), unique=True, nullable=False)


