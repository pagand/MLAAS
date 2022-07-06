#from myapp import app
#if __name__ == "__main__":
#    app.run()
from website import create_app

app = create_app()

if __name__=='__main__':
     app.run()
