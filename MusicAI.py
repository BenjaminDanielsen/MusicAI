import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def splitdata(dataset, genre1, genre2):
    """Splits a dataset into train, test, labels for train, labels for test"""
    genre_one = dataset[dataset["track_genre"] == genre1]
    genre_two = dataset[dataset["track_genre"] == genre2]

    train_genre_one = genre_one[:int(0.7 * len(genre_one))]
    test_genre_one = genre_one[int(0.7 * len(genre_one)):]

    train_genre_two = genre_two[:int(0.7 * len(genre_two))]
    test_genre_two = genre_two[int(0.7 * len(genre_two)):]

    trainset = np.concatenate([train_genre_one, train_genre_two])
    testset = np.concatenate([test_genre_one, test_genre_two])

    np.random.shuffle(trainset)
    np.random.shuffle(testset)

    trainlabel, testlabel = trainset[:,14], testset[:,14]
    trainlabel[trainlabel == genre1] = 1
    trainlabel[trainlabel == genre2] = 0
    testlabel[testlabel == genre1] = 1
    testlabel[testlabel == genre2] = 0
    return trainset[:, 0:13].astype(float), testset[:, 0:13].astype(float), trainlabel, testlabel

def standardization(data):
    """standardize the data with a mean of 0 and a standard deviation of 1"""
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    scaled_data = (data - mean) / std_dev
    return scaled_data

def logic_function(x):
    return 1 / (1 + np.exp(-x + 0.00001))

def train(trainset, labels, weights, learningrate = 0.01):
    
    loss = []
    for i in range(len(trainset)):
        prediction = logic_function(np.dot(weights,trainset[i]))
        error = prediction-labels[i]
        loss.append(abs(error))
        gradient = trainset[i]*error
        weights = weights - learningrate*gradient.T
    
    return weights, sum(loss)/len(trainset)

def test(weights, testset, labels):
    TP, TN, FN, FP = 0,0,0,0
    loss = []
    for i in range(len(testset)):
        a = np.dot(weights, testset[i])
        ab = logic_function(a)
        loss.append(abs(ab - labels[i]))
        if ab > 0.5 and labels[i]:
            TP += 1
        elif ab < 0.5 and not labels[i]:
            TN += 1
        elif ab > 0.5 and not labels[i]:
            FN += 1
        else:
            FP += 1

    confusion_matrix = np.array([[TP,FP], [FN,TN]]) / testset.shape[0]
    return sum(loss)/len(testset), confusion_matrix

if __name__ == "__main__":
    
    # Task 3a
    org_file = pd.read_csv("Dataset.csv", delimiter=",")

    print("Song features in the Dataset:")
    print(org_file.columns)

    genres = org_file["track_genre"].unique()
    print("Different Music Genres:") 
    print(genres)

    total_songs = len(org_file)
    print("Total Number of Songs:", total_songs)

    songs_per_genre = org_file["track_genre"].value_counts()
    print("Number of Songs in Each Genre:")
    print(songs_per_genre)

    if (songs_per_genre == 1000).all():
        print("All genres have 1000 songs.")
    else:
        print("Not all genres have 1000 songs.")

    # Task 3b
    finetuned_dataset = org_file[["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature", "track_genre"]]
    trainset, testset, trainlabel, testlabel = splitdata(finetuned_dataset, "study", "honky-tonk")

    # Task 3c
    plt.scatter(trainset[trainlabel == 0, 2], trainset[trainlabel == 0, 11], label="Class 0", marker="o")
    plt.scatter(trainset[trainlabel == 1, 2], trainset[trainlabel == 1, 11], label="Class 1", marker="x")
    plt.xlabel("danceability")
    plt.ylabel("valence")
    plt.legend()
    plt.show()

    plt.scatter(trainset[trainlabel == 0, 12], trainset[trainlabel == 0, 3], label="Class 0", marker="o")
    plt.scatter(trainset[trainlabel == 1, 12], trainset[trainlabel == 1, 3], label="Class 1", marker="x")
    plt.xlabel("tempo")
    plt.ylabel("energy")
    plt.legend()
    plt.show()

    plt.scatter(trainset[trainlabel == 0, 9], trainset[trainlabel == 0, 6], label="Class 0", marker="o")
    plt.scatter(trainset[trainlabel == 1, 9], trainset[trainlabel == 1, 6], label="Class 1", marker="x")
    plt.xlabel("instrumentalness")
    plt.ylabel("mode")
    plt.legend()
    plt.show()

    trainset, testset = standardization(trainset), standardization(testset)
  
    # Add a bias term
    trainset = np.hstack((trainset, np.ones((trainset.shape[0], 1))))
    testset = np.hstack((testset, np.ones((testset.shape[0], 1))))


    loss_list = []
    loss_test_list = []
    weights = np.random.normal(0,0.1,(14)).T

    # Task 3d
    for i in range(200):
        weights, loss = train(trainset, trainlabel, weights)
        loss_test, confmat = test(weights, testset, testlabel)
        loss_test_list.append(loss_test)
        loss_list.append(loss)

    print(f"The accuracy on the testset is: {confmat[0][0] + confmat[1][1]}")
    print("The confusion matrix is:")
    print(confmat)

    plt.plot(loss_list, label="Loss on trainset")
    plt.plot(loss_test_list, label="Loss on test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Task 3e

    feature_index = [(2, 11, "danceability", "valence"), (12, 3, "tempo", "energy"), (9, 6, "instrumentalness", "mode")]

    for index in feature_index:
        x_min = testset[:, index[0]].min()
        x_max = testset[:, index[0]].max()
        x_values = np.linspace(x_min, x_max, 100)

        a = -weights[index[0]]  # Coefficient of feature_index1
        b = -weights[index[1]]  # Coefficient of feature_index2
        c = -weights[-1]  # Bias term

        # Calculate the decision boundary line (y = mx + c, where m = -a/b)
        m = -a / b
        y_values = m * x_values + c
        plt.scatter(testset[testlabel == 0, index[0]], testset[testlabel == 0, index[1]], label="Class 0", marker="o")
        plt.scatter(testset[testlabel == 1, index[0]], testset[testlabel == 1, index[1]], label="Class 1", marker="x")
        plt.plot(x_values, y_values, label="Decision Boundary")
        plt.xlabel(index[2])
        plt.ylabel(index[3])
        plt.legend()
        plt.title("Test Data and Decision Boundary")
        plt.show()