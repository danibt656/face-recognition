# face recognition

Just a couple of tests for trying to make a facial recognition program that correctly classifies different people's faces

> Before you download, be advised! This repo contains the training image dataset, which is composed of more than 50 MBs of face pictures.

It currently looks like this:

<div style='text-align: center;' align='center'>
    <img style='max-heigth: 120px;' src='./examples/zel.png'/>
</div>

And its performance in terms of predictions' confidence is this:

<div style='text-align: center;' align='center'>
    <img style='max-heigth: 200px;' src='./examples/plot.png'/>
</div>

(as you can see, it gets a little bit confused :^])

## Installation

Before anything can run, you need to create a Python virtual environment & install the dependencies:

```
$ python -m venv env
$ pip install -r requirements.txt
```

Note: for installing the *dlib* library, follow [this tutorial](https://www.youtube.com/watch?v=eaEndTeUiSU&ab_channel=crazzylearners) or [this article](https://pyimagesearch.com/2017/03/27/how-to-install-dlib/).

## Usage

Firstly, run the training program within this repo, locating all faces inside the `faces/` directory, and naming each person's faces folder accordingly (just immitate the given folder structure):

```
$ python faces-train.py
```

Then, go ahead and run the webcam-capture recognizer:

```
$ python cv.py
```

You can get statistic graphs generated after you quit the program if you run it with the `-p`/`--plot` flag:

```
$ python cv.py -p
```

(a full list of options can be obtained with the `-h`/`--help` flag)