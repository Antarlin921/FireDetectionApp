# import library 

from flask import Flask,render_template,request
# from flask import Flask ,render_template,request


from keras.models import load_model
from tensorflow.keras.preprocessing import image

# instace of an app
app=Flask(__name__)
# input modelfile location on your device
model=load_model(r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\Fire Detection\flask_app\firedetection.h5')


@app.route('/')
def hello():
    return render_template("home.html")

# @app.route('/home')
# def homepage():
    return render_template("home.html")

@app.route('/home/detection')
def detectionpage():
    return render_template("detection.html")



#Picture Detection Code Start
@app.route('/blog1',methods=['POST'])
def contact1():  
    piclink= request.form.get('Picture_Link')

    print(piclink)

    #  image path
   
    # read the image
    test_image=image.load_img(piclink,target_size=(64,64))


    # image to array
    test_image=image.img_to_array(test_image)
    test_image=test_image.reshape(1,64,64,3)
    result=model.predict(test_image)
    if result[0][0]==0:
        output="Fire Alert"
    else:
        output="No Fire.False Alarm."
    return render_template('result.html',predicted_text=f'{output}')

#  run the app  
if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)

