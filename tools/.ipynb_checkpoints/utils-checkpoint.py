import pickle


def save_output(output, filename):
    
    with open(filename, 'wb') as file:
        pickle.dump(output, file)
        
    pass


def load_data(filename):

    with open(filename, 'rb') as file:
        data = pickle.load(file)

    return data