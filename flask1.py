from flask import Flask

def create_app():
    app = Flask(__name__)
    app.debug = True
    Bootstrap(app)
    return app

app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

@app.route('/test/')
def test():
    return 'rest'
@app.route('/test1/')
def test1():
    1/0
    return 'rest'
@app.errorhandler(500)
def handle_500(error):
    return str(error), 500
if __name__ == '__main__':
    app.run()