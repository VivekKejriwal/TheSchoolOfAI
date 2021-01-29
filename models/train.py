{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "train.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOyl8eCs5mRAgTBu8+4VR4u",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/VivekKejriwal/TheSchoolOfAI/blob/main/models/train.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9xQZKpTzlghc"
      },
      "source": [
        "from tqdm import tqdm\r\n",
        "\r\n",
        "def train(model,device,train_loader,criterion,optimizer,epoch):\r\n",
        "  model.train()\r\n",
        "  pbar = tqdm(train_loader)\r\n",
        "  correct = 0\r\n",
        "  processed = 0\r\n",
        "\r\n",
        "  for batch_idx,(data, target) in enumerate(pbar):\r\n",
        "    data, target = data.to(device), target.to(device)\r\n",
        "\r\n",
        "    optimizer.zero_grad()\r\n",
        "\r\n",
        "    y_pred = model(data)\r\n",
        "\r\n",
        "    loss = criterion(y_pred,target)\r\n",
        "\r\n",
        "    loss.backward()\r\n",
        "    optimizer.step()\r\n",
        "\r\n",
        "    pred= y_pred.argmax(dim=1,keepdim=True)\r\n",
        "    correct+= pred.eq(target.view_as(pred)).sum().item()\r\n",
        "    processed+= len(data)\r\n",
        "\r\n",
        "    pbar_str = f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'\r\n",
        "    pbar.set_description(desc= pbar_str)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}