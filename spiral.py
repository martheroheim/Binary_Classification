import json

if __name__ == '__main__':

    stream = open("config_spiral.yaml", 'r')
    dictionary = yaml.load(stream)
    for key, value in dictionary.items():
        print (key + " : " + str(value))