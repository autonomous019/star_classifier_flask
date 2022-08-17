# star_classifier_flask

star classifier using xgboost, web application uses Flask.  Classify stars by type, input Temperature, Luminosity, Radius and Absolute Magnitude and returns a numerical type of the star, used by Astronomers. 


Work flow:
    stars_model.py - creates a model from trained data using xgb.Classifier
    app.py - runs the application on a web server, routes set the user interface in this file and loads the precompiled model in .pkl format using Pickle serialization. Flask is the web app python framework
    templates/index.html is the html code, user interface, that is called by app.py
    model.json is not used but I generate it just in case I choose to use it some day. 
    
    if you don't want to generate a model on your own, just run app.py in a vitrual environment in PyCharm, when setting up PyCharm be sure to add Flask when creating the virtual environment. 
