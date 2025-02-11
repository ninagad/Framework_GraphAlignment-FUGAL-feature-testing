import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    path = os.path.join(os.path.dirname(__file__), 'runs\\28\\res\\acc.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise this fails with Errno 13 permission denied.
    df = pd.read_excel(path)

    df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

    # Fill NaN values with the previous row values
    df['Features'] = df['Features'].ffill()

    df['mean'] = df.iloc[:,2:].mean(axis=1)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Loop through unique features and plot each one
    for feature in df['Features'].unique():
        subset = df[df['Features'] == feature]
        plt.plot(subset['Noise-level'], subset['mean'], marker='o', label=feature)

    # Customize plot
    plt.ylim(-0.1,1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Mean')
    plt.title('Mean vs. Noise-level for Different Features')
    plt.legend(title='Features')
    plt.grid(True)

    path = os.path.join(os.path.dirname(__file__), 'runs\\28\\res\\acc.svg')
    plt.savefig(path)

# Using the special variable
# __name__
if __name__=="__main__":
    main()


