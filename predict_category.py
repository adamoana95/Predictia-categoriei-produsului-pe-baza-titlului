{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPNkhyM14awOkWsivQmNDsW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
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
        "<a href=\"https://colab.research.google.com/github/adamoana95/Predictia-categoriei-produsului-pe-baza-titlului/blob/main/predict_category.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "24JaeXke8qRU",
        "outputId": "2786a8a1-9c7b-4d3f-a1a5-5abf79545a4d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model loaded successfully!\n",
            "Type 'exit' at any point to stop.\n",
            "\n",
            "Enter product title: iphone 7 32gb gold,4,3,Apple iPhone 7 32GB\n",
            "Predicted category: Mobile Phones\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: kenwood k20mss15 solo\n",
            "Predicted category: Microwaves\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: smeg sbs8004po\n",
            "Predicted category: Dishwashers\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: bosch serie 4 kgv39vl31g\n",
            "Predicted category: Fridge Freezers\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: bosch wap28390gb 8kg 1400 spin\n",
            "Predicted category: Washing Machines\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: olympus e m10 mark iii geh use silber\n",
            "Predicted category: Digital Cameras\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: smeg sbs8004po\n",
            "Predicted category: Dishwashers\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: Bosch KGV58VLEAS, 503 l, Low Frost\n",
            "Predicted category: Fridge Freezers\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: Samsung MS23K3513, 23 l, Digital\n",
            "Predicted category: Digital Cameras\n",
            "\n",
            "----------------------------------------\n",
            "Enter product title: exit\n",
            "Exiting...\n"
          ]
        }
      ],
      "source": [
        "import re\n",
        "import joblib\n",
        "import pandas as pd\n",
        "\n",
        "# Load the saved model\n",
        "model = joblib.load(\"model/final_model.pkl\")\n",
        "\n",
        "print(\"Model loaded successfully!\")\n",
        "print(\"Type 'exit' at any point to stop.\\n\")\n",
        "\n",
        "while True:\n",
        "\n",
        "    title = input(\"Enter product title: \")\n",
        "\n",
        "    if title.lower() == \"exit\":\n",
        "        print(\"Exiting...\")\n",
        "        break\n",
        "\n",
        "    # Calculate features\n",
        "    title_char_count = len(title)\n",
        "    title_word_count = len(title.split())\n",
        "    has_number = int(bool(re.search(r'\\d', title)))\n",
        "    number_count = len(re.findall(r'\\d', title))\n",
        "    special_char_count = len(re.findall(r'[^a-zA-Z0-9\\s]', title))\n",
        "    has_special_char = int(special_char_count > 0)\n",
        "    max_word_length = max(len(word) for word in title.split())\n",
        "\n",
        "    # Create DataFrame\n",
        "    user_input = pd.DataFrame([{\n",
        "        \"product title\": title,\n",
        "        \"title_char_count\": title_char_count,\n",
        "        \"title_word_count\": title_word_count,\n",
        "        \"has_number\": has_number,\n",
        "        \"number_count\": number_count,\n",
        "        \"special_char_count\": special_char_count,\n",
        "        \"has_special_char\": has_special_char,\n",
        "        \"max_word_length\": max_word_length\n",
        "    }])\n",
        "\n",
        "    # Predict category\n",
        "    prediction = model.predict(user_input)[0]\n",
        "\n",
        "    print(f\"Predicted category: {prediction}\\n\")\n",
        "    print(\"-\" * 40)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pwd\n",
        "!find /content -name \"final_model.pkl\""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nItpVGdF-aGK",
        "outputId": "59438185-8832-4ad2-b215-2bea5415de14"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/adamoana95/Predictia-categoriei-produsului-pe-baza-titlului.git"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aKxKijTC-uMn",
        "outputId": "a4c08cb7-5872-4a02-a76a-9f756191a169"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'Predictia-categoriei-produsului-pe-baza-titlului'...\n",
            "remote: Enumerating objects: 62, done.\u001b[K\n",
            "remote: Counting objects: 100% (62/62), done.\u001b[K\n",
            "remote: Compressing objects: 100% (56/56), done.\u001b[K\n",
            "remote: Total 62 (delta 19), reused 9 (delta 0), pack-reused 0 (from 0)\u001b[K\n",
            "Receiving objects: 100% (62/62), 3.13 MiB | 22.56 MiB/s, done.\n",
            "Resolving deltas: 100% (19/19), done.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /content"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wf2Ov-CH-7cs",
        "outputId": "53dc9c42-524d-4304-a68e-0788d0a1e288"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Predictia-categoriei-produsului-pe-baza-titlului  sample_data\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%cd /content/Predictia-categoriei-produsului-pe-baza-titlului"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7UJmS3Ml_GZ7",
        "outputId": "b46e7d79-2f10-457c-8dd7-ff52f2cb5fb8"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/Predictia-categoriei-produsului-pe-baza-titlului\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fPxSdg1__K6j",
        "outputId": "f0ad91d7-24ee-47e2-9ee8-af2cc9f0a53d"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "data  model  notebooks\tpredict_category.py  README.md\ttrain_model.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls model"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5tI1g9FB_M0h",
        "outputId": "7d5684fe-8130-49c9-95f9-d9b525ee11a9"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "final_model.pkl\n"
          ]
        }
      ]
    }
  ]
}