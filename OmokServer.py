import os
import time

import tensorflow as tf
from flask import Flask, request
from flask.json import jsonify

from OmokTrainDeep import OmokEnvironment

app = Flask(__name__, static_url_path='/static')

# 텐서플로우 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 세이브 설정
saver = tf.train.Saver()

gridSize = 8

# 모델 로드
if (os.path.isfile(os.getcwd() + "/OmokModelDeep.ckpt.index") == True):
    saver.restore(sess, os.getcwd() + "/OmokModelDeep.ckpt")
    print('Saved model is loaded!')

# 환경 인스턴스 생성
env = OmokEnvironment(gridSize)

STONE_PLAYER1 = 1
STONE_PLAYER2 = 2

app.beforePlayer = 0


@app.route('/<path:currentPlayer>', methods=['GET'])
def get(currentPlayer):
    # currentPlayer = int(currentPlayer)
    # print(currentPlayer)

    time.sleep(1)

    if app.beforePlayer == currentPlayer:
        return jsonify({
            'success': False
        })

    app.beforePlayer = currentPlayer

    if (currentPlayer == STONE_PLAYER1):
        currentState = env.getState()
    else:
        currentState = env.getStateInverse()

    action = env.getAction(sess, currentState)
    print(action)

    nextState, reward, gameOver = env.act(currentPlayer, action)

    print(currentState)
    print(nextState)

    changedPosition = [i for i in range(len(currentState[0])) if currentState[0][i] != nextState[0][i]]

    print(changedPosition)
    position_x = int(changedPosition[0] / 8)
    position_y = changedPosition[0] % 8

    return jsonify({
        'gameOver': gameOver,
        'position': changedPosition,
        'position_x': position_x,
        'position_y': position_y,
        'success': True
    })


@app.route('/<path:currentPlayer>', methods=['POST'])
def post(currentPlayer):
    # currentPlayer = int(currentPlayer)
    if app.beforePlayer == currentPlayer:
        return jsonify({
            'success': False
        })

    if (currentPlayer == STONE_PLAYER1):
        currentState = env.getState()
    else:
        currentState = env.getStateInverse()

    nextState, reward, gameOver = env.act(currentPlayer, request.args['action'])

    changedPosition = [i for i in range(len(currentState[0])) if currentState[0][i] != nextState[0][i]]
    position_x = int(changedPosition[0] / 8)
    position_y = changedPosition[0] % 8

    return jsonify({
        'gameOver': gameOver,
        'position': changedPosition,
        'position_x': position_x,
        'position_y': position_y,
        'success': True
    })


if __name__ == '__main__':
    app.run(port=5000)
