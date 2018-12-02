from flask import Flask, request, jsonify
from flask import render_template
from models.PhiNet import *
import os
app = Flask(__name__)
Users = {}


@app.route('/Welcome')
def Welcome():
    return render_template('Welcome.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():

    user = request.form['UserId']
    if user in Users:
        image = request.files['Signature']
        print("Score: "+str('{:0.3f}'.format(EC_dist(PhiNet(PreProcess(image)), Users[user]).item())))
        return jsonify("Score: "+str('{:0.3f}'.format(EC_dist(PhiNet(PreProcess(image)), Users[user]).item())))

    else:
        print("User isn't present!")
        return jsonify("User isn't present!")


@app.route('/register', methods=['POST'])
def register():
    user = request.form['UserId']
    image = request.files['Signature']

    if user in Users:
        print("User is already registered")
        return jsonify("User is already registered!")

    Users[user] = PhiNet(PreProcess(image)).data

    # image.save('/home/tug/WS/PhiNet/db/'+image.filename)
    print("User: "+str(user)+" has been successfully registered")
    return jsonify("User: "+str(user)+" has been successfully registered")


@app.route('/check', methods=['GET'])
def check():
    print({i: user for i, user in enumerate(Users.keys)})
    return jsonify({i: user for i, user in enumerate(Users.keys)})


if __name__ == '__main__':
    host = os.popen("hostname -I").read().split(" ")[0]
    PhiNet = ConvNet()
    PhiNet.load_state_dict(torch.load("models/phinet_siamese_theone_dict (3).stdt", map_location="cpu"))
    # PhiNet.eval()
    #PhiNet = torch.load("models/phinet_siamese_theone (3).stdt", map_location="cpu")["net"]
    #torch.save(PhiNet.state_dict(), "models/phinet_siamese_theone_dict (3).stdt")
    PhiNet.eval()
    os.system("clear")
    print(" * Model has been imported....")
    print(" * PhiNet is running....")
    app.run(host=host, port="5000")
