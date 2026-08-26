{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO57Ogkx9UQIXucPc31v00k",
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
      "execution_count": null,
      "metadata": {
        "id": "24JaeXke8qRU"
      },
      "outputs": [],
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
        "        \"title char count\": title_char_count,\n",
        "        \"title word count\": title_word_count,\n",
        "        \"has number\": has_number,\n",
        "        \"number count\": number_count,\n",
        "        \"special char count\": special_char_count,\n",
        "        \"has special char\": has_special_char,\n",
        "        \"max word length\": max_word_length\n",
        "    }])\n",
        "\n",
        "    # Predict category\n",
        "    prediction = model.predict(user_input)[0]\n",
        "\n",
        "    print(f\"Predicted category: {prediction}\\n\")\n",
        "    print(\"-\" * 40)"
      ]
    }
  ]
}