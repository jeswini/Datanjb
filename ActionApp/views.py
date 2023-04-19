from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import io
import base64

# Create your views here.

@csrf_exempt
def actionApi(request):

    if request.method == 'POST':
        
        message = ""

        data = JSONParser().parse(request)

        actionName = data["ActionName"]

        #merge action
        if actionName == "Merge":
            fileList = data["fileList"]
            mergeColumn = data["mergeColumn"]
            dataframes = []
            for fileName in fileList:
                dataFile = pd.read_csv(fileName)
                dataframes.append(dataFile)
            
            mergeData = dataframes[0]

            for i in range(1,len(fileList)):
                mergeData = mergeData.merge(dataframes[i],on=mergeColumn,how='outer')
            
            data = []
            for rows in mergeData:
                data.append(rows)
            
            result = mergeData.to_json(orient="records")
            message = json.loads(result)
        
        #sort action
        if actionName == "Sort":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            sortColumnNames = data["sortColumnNames"]
            sortedData = dataFile.sort_values(by=sortColumnNames,kind='mergesort')
            
            result = sortedData.to_json(orient="records")
            message = json.loads(result)

        #projection action
        if actionName == "Projection":
            jsonData = data["jsonData"]
            projectColumns = data["projectColumns"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            selectedData = pd.DataFrame(dataFile,columns=projectColumns)
            
            result = selectedData.to_json(orient="records")
            message = json.loads(result) 

        #encode action
        if actionName == "Encode":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            encodingColumns = data["encodingColumns"]
            labelencoder = LabelEncoder()

            for i in encodingColumns: 
                dataFile[i] =  labelencoder.fit_transform(dataFile[i])
            
            result = dataFile.to_json(orient="records")
            message = json.loads(result)
        
        #coorelation action
        if actionName == "Correlation":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
                    
            outputFileName = data["outputFileName"]
            pearsoncorr = dataFile.corr(method='pearson')
            
            plt.figure()

            f, ax = plt.subplots(figsize =(15, 15))
            sb.heatmap(pearsoncorr, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

            #sb.pairplot(dataFile)

            plt.savefig(outputFileName)

            result = dataFile.to_json(orient="records")

            stringIObytes = io.BytesIO()
            plt.savefig(stringIObytes, format='jpg')
            stringIObytes.seek(0)
            correlation_matrix_image = str(base64.b64encode(stringIObytes.read()))

            #response['jsonData'] = result
            #response['encodedString'] = my_base64_jpgData

            sol = round((pearsoncorr.where(np.triu(np.ones(pearsoncorr.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=True)),2)
            data = {}
            i=0
            beg = 0
            end = 0
            for index, value in sol.items():
                if(i==0):
                    data[index] = value
                    beg = value
                if(i == len(sol)-1):
                    data[index] = value
                    end = value
                i+=1

            mid = (beg+end)/2
            for index, value in sol.items():
                if(value >= mid):
                    data[index] = value
                    break
            
            x = data.keys()
            j=0
            for i in x:
                plt.figure()
                if (j == 0):
                    sb.regplot(x=dataFile[i[0]], y=dataFile[i[1]])
                    stringIObytes = io.BytesIO()
                    plt.title("Negative Correlation")
                    plt.savefig(stringIObytes, format='jpg')
                    stringIObytes.seek(0)
                    lowCorrelationEncodedString = str(base64.b64encode(stringIObytes.read()))
                if (j == 1):
                    sb.regplot(x=dataFile[i[0]], y=dataFile[i[1]])
                    stringIObytes = io.BytesIO()
                    plt.title("Positive Correlation")
                    plt.savefig(stringIObytes, format='jpg')
                    stringIObytes.seek(0)
                    highCorrelationEncodedString = str(base64.b64encode(stringIObytes.read()))
                if (j == 2):
                    sb.regplot(x=dataFile[i[0]], y=dataFile[i[1]])
                    stringIObytes = io.BytesIO()
                    plt.title("Correlation?")
                    plt.savefig(stringIObytes, format='jpg')
                    stringIObytes.seek(0)
                    mediumCorrelationEncodedString = str(base64.b64encode(stringIObytes.read()))
                j+=1

            response = {
                "jsonData" : result,
                "Correlation_matrix_encodedString" : correlation_matrix_image,
                "Low_Correaltion_graph_encodedString" : lowCorrelationEncodedString,
                "Medium_Correaltion_graph_encodedString" : highCorrelationEncodedString,
                "High_Correaltion_graph_encodedString" : mediumCorrelationEncodedString
            }
            message = response

        #analysis
        if actionName == "Decision Tree":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            x_columns =  ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex"]
            y_columns = ["salary"]
            x_data =  pd.DataFrame(dataFile,columns=x_columns)

            y_data =  pd.DataFrame(dataFile,columns=y_columns)

            x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3)
            
            sc = StandardScaler()

            X_train = sc.fit_transform(x_training_data)
            X_test = sc.transform(x_test_data)

            classifer = DecisionTreeClassifier(criterion='entropy', random_state=0)
            classifer.fit(X_train,y_training_data)
            y_pred = classifer.predict(X_test)

            accuracy=accuracy_score(y_test_data,y_pred)
            format_float = "{:.2f}".format(accuracy*100)

            acc = str(format_float) + "%"
            response = { "accuracy" : acc }
            message = response


        if actionName == "UPPERCASE":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            
            dataFile_columns = dataFile.columns
            dataFile_dtypes = dataFile.dtypes

            for i in range(0,len(dataFile_columns)-1):
                if(dataFile_dtypes[i] == 'object'):
                    dataFile[dataFile_columns[i]] = dataFile[dataFile_columns[i]].str.upper()
            
            result = dataFile.to_json(orient="records")
            message = json.loads(result)

        if actionName == "lowercase":
            jsonData = data["jsonData"]
            json_object = json.loads(jsonData)
            dataFile = pd.json_normalize(json_object)
            
            dataFile_columns = dataFile.columns
            dataFile_dtypes = dataFile.dtypes

            for i in range(0,len(dataFile_columns)-1):
                if(dataFile_dtypes[i] == 'object'):
                    dataFile[dataFile_columns[i]] = dataFile[dataFile_columns[i]].str.lower()
            
            result = dataFile.to_json(orient="records")
            message = json.loads(result) 

    return JsonResponse(message,safe=False)


