{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMfcgn0DHrK3MJh0kDxah0w",
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
        "<a href=\"https://colab.research.google.com/github/adamoana95/Predictia-categoriei-produsului-pe-baza-titlului/blob/main/train_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ganhdemD3Wk2",
        "outputId": "ec86927e-be1f-4176-a567-3115dd779ffe"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model trained and saved successfully!\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import re\n",
        "import joblib\n",
        "import pandas as pd\n",
        "\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.svm import LinearSVC\n",
        "\n",
        "\n",
        "# Load dataset from GitHub\n",
        "url = \"https://raw.githubusercontent.com/adamoana95/Predictia-categoriei-produsului-pe-baza-titlului/main/data/products.csv\"\n",
        "\n",
        "df = pd.read_csv(url)\n",
        "\n",
        "# Clean column names\n",
        "df.columns = df.columns.str.strip().str.replace('_', ' ', regex=False).str.lower()\n",
        "\n",
        "# Remove missing values\n",
        "df = df.dropna()\n",
        "\n",
        "\n",
        "# Standardize category names\n",
        "category_mapping = {\n",
        "    'fridge': 'Fridges',\n",
        "    'CPU': 'CPUs',\n",
        "    'Mobile Phone': 'Mobile Phones'\n",
        "}\n",
        "\n",
        "df['category label'] = df['category label'].replace(category_mapping)\n",
        "\n",
        "\n",
        "# Feature engineering\n",
        "df['title_char_count'] = df['product title'].str.len()\n",
        "\n",
        "df['title_word_count'] = df['product title'].str.split().str.len()\n",
        "\n",
        "df['has_number'] = df['product title'].str.contains(\n",
        "    r'\\d', regex=True\n",
        ").astype(int)\n",
        "\n",
        "df['number_count'] = df['product title'].str.count(r'\\d')\n",
        "\n",
        "df['special_char_count'] = df['product title'].apply(\n",
        "    lambda x: len(re.findall(r'[^a-zA-Z0-9\\s]', x))\n",
        ")\n",
        "\n",
        "df['has_special_char'] = (\n",
        "    df['special_char_count'] > 0\n",
        ").astype(int)\n",
        "\n",
        "df['max_word_length'] = df['product title'].str.split().apply(\n",
        "    lambda words: max(len(word) for word in words)\n",
        ")\n",
        "\n",
        "\n",
        "# Remove columns that are not used for prediction\n",
        "columns_to_drop = [\n",
        "    'product id',\n",
        "    'merchant id',\n",
        "    'product code',\n",
        "    'number of views',\n",
        "    'merchant rating',\n",
        "    'listing date'\n",
        "]\n",
        "\n",
        "df = df.drop(\n",
        "    columns=[col for col in columns_to_drop if col in df.columns]\n",
        ")\n",
        "\n",
        "\n",
        "# Define features and target\n",
        "numeric_features = [\n",
        "    'title_char_count',\n",
        "    'title_word_count',\n",
        "    'has_number',\n",
        "    'number_count',\n",
        "    'special_char_count',\n",
        "    'has_special_char',\n",
        "    'max_word_length'\n",
        "]\n",
        "\n",
        "X = df.drop(columns=['category label'])\n",
        "y = df['category label']\n",
        "\n",
        "\n",
        "# Preprocessing\n",
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        (\n",
        "            'title',\n",
        "            TfidfVectorizer(\n",
        "                lowercase=True,\n",
        "                ngram_range=(1, 2),\n",
        "                max_features=50000\n",
        "            ),\n",
        "            'product title'\n",
        "        ),\n",
        "        (\n",
        "            'numeric',\n",
        "            MinMaxScaler(),\n",
        "            numeric_features\n",
        "        )\n",
        "    ]\n",
        ")\n",
        "\n",
        "\n",
        "# Final model\n",
        "pipeline = Pipeline([\n",
        "    ('preprocessing', preprocessor),\n",
        "    ('classifier', LinearSVC())\n",
        "])\n",
        "\n",
        "\n",
        "# Train\n",
        "pipeline.fit(X, y)\n",
        "\n",
        "\n",
        "# Save model\n",
        "os.makedirs('model', exist_ok=True)\n",
        "\n",
        "joblib.dump(\n",
        "    pipeline,\n",
        "    'model/final_model.pkl'\n",
        ")\n",
        "\n",
        "print(\"Model trained and saved successfully!\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git add train_model.py\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a2u8s8Vx6jB7",
        "outputId": "41ec7d95-5193-4f6e-ec9d-5c20e9f9a95d"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "fatal: not a git repository (or any of the parent directories): .git\n"
          ]
        }
      ]
    }
  ]
}