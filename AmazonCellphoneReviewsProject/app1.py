from flask import Flask,render_template,request,url_for
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
graph=tf.compat.v1.get_default_graph()
#global graph
#graph=tf.compat.v1.Graph()'''

with open(r'count_vec1.pkl','rb') as file:
    cv=pickle.load(file)


app=Flask(__name__)
#Routing to the HTML Page:
@app.route('/')
def home():
    return render_template('home.html')
 #Showcasing prediction on UI

@app.route('/tpredict',methods=['GET','POST'])
def page2():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method =='POST':
        topic = request.form['review']
        print("Hey " +topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
     
      
        with graph.as_default():
            cla = load_model('AmazonReview_model_saved1.h5')
            cla.compile(optimizer='adam',loss='binary_crossentropy') 
            y_pred =cla.predict(topic.toarray())
            print("pred is "+str(y_pred))
        if(y_pred >0.5):
            topic ="Positive Review"
        else:
            topic ="Negative Review"
        return render_template('home.html',ypred=topic)
if  __name__ =="__main__":
    app.run(debug=True)
