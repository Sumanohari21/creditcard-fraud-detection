import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
selected = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload & Generate", "🚨 Train Model"])

# 🏠 Home Page
if selected == "🏠 Home":
    st.title("💳 Privacy-Preserving Credit Card Fraud Detection with GANs")
    st.markdown("""
    This project uses GANs to generate synthetic data for credit card fraud detection.
    - Upload your dataset
    - Generate synthetic samples
    - Train fraud detection model
    """)


# 📤 Upload & Generate Page
elif selected == "📤 Upload & Generate":
    st.header("📤 Upload Dataset & Generate Synthetic Data")

    uploaded_file = st.file_uploader("Upload your creditcard.csv", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Preprocessing
        scaler = MinMaxScaler()
        df['normAmount'] = scaler.fit_transform(df[['Amount']])
        df = df.drop(['Time', 'Amount'], axis=1)

        fraud = df[df['Class'] == 1]
        non_fraud = df[df['Class'] == 0].sample(n=len(fraud)*2, random_state=1)
        df_balanced = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

        X_real = df_balanced.drop('Class', axis=1)
        y_real = df_balanced['Class']

        st.session_state['X_real'] = X_real
        st.session_state['y_real'] = y_real

        st.success("✅ Preprocessing complete and data balanced.")

        if st.button("🧠 Generate Synthetic Data"):
            generator = Sequential([
                Dense(64, activation='relu', input_dim=100),
                Dense(X_real.shape[1], activation='sigmoid')
            ])

            discriminator = Sequential([
                Dense(64, activation='relu', input_dim=X_real.shape[1]),
                Dense(1, activation='sigmoid')
            ])
            discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            discriminator.trainable = False
            gan = Sequential([generator, discriminator])
            gan.compile(optimizer='adam', loss='binary_crossentropy')

            for _ in range(300):
                noise = np.random.normal(0, 1, (32, 100))
                fake_data = generator.predict(noise)
                real_data = X_real.sample(32)

                discriminator.trainable = True
                discriminator.train_on_batch(real_data, np.ones((32, 1)))
                discriminator.train_on_batch(fake_data, np.zeros((32, 1)))

                discriminator.trainable = False
                gan.train_on_batch(noise, np.ones((32, 1)))

            noise = np.random.normal(0, 1, (500, 100))
            synthetic = generator.predict(noise)
            synthetic_df = pd.DataFrame(synthetic, columns=X_real.columns)

            st.session_state['synthetic_df'] = synthetic_df

            st.write("### 🔍 Preview of Synthetic Data")
            st.dataframe(synthetic_df.head())

            csv = synthetic_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Synthetic Data", csv, "synthetic_data.csv", "text/csv")

# 🚨 Train Model Page
elif selected == "🚨 Train Model":
    st.header("🚨 Train Fraud Detection Model")

    if 'X_real' in st.session_state and 'synthetic_df' in st.session_state:
        X_real = st.session_state['X_real']
        y_real = st.session_state['y_real']
        synthetic_df = st.session_state['synthetic_df']

        if st.button("🚨 Train Model"):
            synthetic_df['Class'] = np.random.choice([0, 1], len(synthetic_df))
            X_combined = pd.concat([X_real, synthetic_df.drop('Class', axis=1)])
            y_combined = pd.concat([y_real, synthetic_df['Class']])

            X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = clf.score(X_test, y_test)
            st.success(f"🎯 Model Accuracy: {acc * 100:.2f}%")

            st.write("### 📋 Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

            st.write("### 📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
    else:
        st.warning("⚠️ You need to generate synthetic data first.")
