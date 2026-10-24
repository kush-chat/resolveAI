import pandas as pd
from sklearn.preprocessing import StandardScaler #introduced at step 29
from sklearn.compose import ColumnTransformer #introduced at step 30

#introduced at step 31
import numpy as np
from scipy.sparse import issparse

#introduced at step 33
from sklearn.linear_model import LogisticRegression

#introduced at step 35
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

#introduced at step 38
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone

#introduced at step 39
from sklearn.naive_bayes import GaussianNB

#introduced at step 41
from sklearn.preprocessing import FunctionTransformer
#introduced at step 42
from sklearn.svm import LinearSVC, SVC
#introduced at step 48
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LinearRegression #49
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #50
from sklearn.model_selection import KFold #51
from sklearn.dummy import DummyRegressor #54
from sklearn.svm import LinearSVR #56
from sklearn.compose import TransformedTargetRegressor #63
from sklearn.linear_model import Ridge #72


#step 1: load oirginal incident dataset 
data = pd.read_csv("data/incident_event_log.csv")

#initial verification
print(f"Inital Total rows: {len(data)}")
print(f"Initial Unique incidents: {data['number'].nunique()}")

#inspect few orginal time stamp values before converting them
print("timestamp values: \n")
print(data[["number", "opened_at", "sys_updated_at", "incident_state"]].head(10))


# Step 2: Convert timestamp strings into datetime values
data["opened_at"] = pd.to_datetime(data["opened_at"], dayfirst=True)
data["sys_updated_at"] = pd.to_datetime(data["sys_updated_at"], dayfirst=True)

# Sort records chronologically by their update timestamp
data = data.sort_values(by="sys_updated_at", ascending=True, kind="stable")

# Keep only the earliest available record for each incident
data = data.drop_duplicates(subset="number", keep="first")

# Reset row indexes after removing duplicates
data = data.reset_index(drop=True)

# Verify the resulting dataset
print("\nTotal rows:", len(data))
print("Unique incidents:", data["number"].nunique())

# Examine the earliest recorded state of each incident
print("\nIncident state distribution:")
print(data["incident_state"].value_counts(dropna=False))

print("\nEarliest recorded events:")
print(data[["number", "opened_at", "sys_updated_at", "incident_state"]].head(10))

# Step 3: Investigate incident timestamps (since some data are updated before ticket created/opened)

# Calculate elapsed time between opening and the first recorded update.
data["minutes_since_open"] = (data["sys_updated_at"] - data["opened_at"]).dt.total_seconds()/60

#Identify incidents updated before their opening timestamp.
negative_time =  data[data["minutes_since_open"] < 0]
print("\nIncidents with negative elapsed time:", len(negative_time))
print(negative_time[["number", "opened_at", "sys_updated_at", "minutes_since_open"]].head(5))

#Find incidents whose first update was within 60 minutes of opening.
within_one_hour = data[(data["minutes_since_open"] >= 0) & (data["minutes_since_open"] <= 60)]
print("\nIncidents updated within one hour:", len(within_one_hour))

#Investigate incidents whose earliest recorded state is Resolved.
resolved_incidents = data[data["incident_state"] == "Resolved"]
print("\nInitially resolved incidents:", len(resolved_incidents))

#Check how many resolved incidents have an early update.
resolved_within_hour = resolved_incidents[(resolved_incidents["minutes_since_open"] >= 0) &
                                          (resolved_incidents["minutes_since_open"] <= 60)]
print("\nResolved incidents updated within one hour:", len(resolved_within_hour))

#Inspect examples of incidents already marked Resolved.
print("\nSample resolved incidents:")
print(resolved_incidents[["number", "opened_at", "sys_updated_at", "minutes_since_open"]].head(5))

#step 4: Investigate the resolution timestamps

#Convert the resolution timestamp to datetime.
data["resolved_at"] = pd.to_datetime(data["resolved_at"], dayfirst=True, errors="coerce")

# Select incidents whose earliest recorded state is Resolved.
resolved_incidents = data[data["incident_state"] == "Resolved"].copy()

# Calculate the actual resolution duration in minutes.
resolved_incidents["resolution_minutes"] = (resolved_incidents["resolved_at"] - 
                                            resolved_incidents["opened_at"]).dt.total_seconds() / 60

# Examine the resolution durations.
print("\nInitially resolved incidents: ", len(resolved_incidents))
print("\nResolved in zero minutes:: ", (resolved_incidents["resolution_minutes"] == 0).sum())
print("Resolved in more than zero minutes: ", (resolved_incidents["resolution_minutes"] > 0).sum())
print("Negative resolution duration: ", (resolved_incidents["resolution_minutes"] < 0).sum())
print("Missing resolution duration: ", resolved_incidents["resolution_minutes"].isna().sum())

# Inspect actual timestamps for the first 10 incidents.
print("\nSample resolved incidents:")
print(
    resolved_incidents[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "resolved_at",
            "resolution_minutes"
        ]
    ].head(10).to_string(index=False)
)

# Investigate incidents whose earliest recorded state is New.
new_incidents = data[data["incident_state"] == "New"]

print("\nInitially New incidents:", len(new_incidents))
print("New incidents with a resolution timestamp:",new_incidents["resolved_at"].notna().sum())
print("New incidents with a future resolution timestamp:",(new_incidents["resolved_at"] > new_incidents["opened_at"]).sum())

#from the observation, found many missing resolution duration
# Step 5: Recover missing resolution timestamps from incident history

# Load the original dataset containing all event records.
history = pd.read_csv("data/incident_event_log.csv")

# Convert resolution timestamps into datetime values.
history["resolved_at"] = pd.to_datetime(history["resolved_at"], dayfirst=True, errors="coerce")

# Group all historical records by incident ID.
# max() returns the latest available resolution timestamp for each incident.
resolution_history = history.groupby("number")["resolved_at"].max()

# Match each ticket in our prepared dataset to its historical resolution timestamp.
data["history_resolved_at"] = data["number"].map(resolution_history)

# Select incidents initially marked Resolved with a missing resolution timestamp.
missing_resolution = data[(data["incident_state"] == "Resolved") & (data["resolved_at"].isna())].copy()

print("\nInitially resolved incidents with missing timestamps: ", len(missing_resolution))
print(
    "\nRecovered resolution timestamps:",
    missing_resolution["history_resolved_at"].notna().sum()
)
print(
    "\nStill missing resolution timestamps:",
    missing_resolution["history_resolved_at"].isna().sum()
)
print("\nSample incidents:")

print(
    missing_resolution[
        ["number", "resolved_at", "history_resolved_at"]
    ].head(5).to_string(index=False)
)

# Step 6: Validate resolution durations across all incidents

# Calculate resolution duration in hours.
# The duration is our future regression target (y).
data["resolution_hours"] = (data["history_resolved_at"] - data["opened_at"]).dt.total_seconds() /3600

# Identify valid, negative, and missing durations.
valid_resolution = data["resolution_hours"] > 0
negative_resolution = data["resolution_hours"] < 0
missing_resolution = data["resolution_hours"].isna()

print("\nValid resolution durations:", valid_resolution.sum())
print("Negative resolution durations:", negative_resolution.sum())
print("Missing resolution durations:", missing_resolution.sum())

# Check target availability for each initial incident state.
# This is a data-quality report, not a list of model input features.
data["valid_resolution"] = valid_resolution
data["missing_resolution"] = missing_resolution

print("\nResolution availability by initial state:")
print(
    data.groupby("incident_state")[
        ["valid_resolution", "missing_resolution"]
    ].sum()
)

# Examine five incidents with valid resolution durations.
print("\nSample incidents with valid resolution durations:")
print(
    data.loc[
        valid_resolution,
        [
            "number",
            "opened_at",
            "history_resolved_at",
            "resolution_hours"
        ]
    ].head(5).to_string(index=False)
)

# Step 7: Validate every resolution duration
# Count positive, zero, negative, and missing durations.
positive_count = (data["resolution_hours"] > 0).sum()
zero_count = (data["resolution_hours"] == 0).sum()
negative_count = (data["resolution_hours"] < 0).sum()
missing_count = (data["resolution_hours"].isna().sum())

print("\nPositive durations:", positive_count)
print("Zero durations:", zero_count)
print("Negative durations:", negative_count)
print("Missing durations:", missing_count)

# Verify that every incident belongs to one of these categories.
total = (positive_count + negative_count + zero_count + missing_count)
print("\nTotal accounted incidents:", total)
print("Actual dataset rows:", len(data))

# Inspect incidents with a resolution duration of zero hours.
zero_incidents = data[data["resolution_hours"] == 0]
print("\nSample zero-hour incidents:")
print(
    zero_incidents[
        [
            "number",
            "opened_at",
            "history_resolved_at",
            "incident_state",
            "resolution_hours"
        ]
    ].head(5).to_string(index=False)
)

#step 8: Verify resolution timing relative to input snapshot

# Keep incidents that are not initially Resolved and whose earliest update is not before opening.
candidates = data[(data["incident_state"]  != "Resolved") & (data["minutes_since_open"] >= 0)].copy()

# Resolution occurred after our input snapshot.
future_resolution = (candidates["history_resolved_at"] > candidates["sys_updated_at"])

# Historical resolution timestamp is unavailable.
missing_resolution = (candidates["history_resolved_at"].isna())

# Resolution occurred at or before our snapshot.
already_resolved = (candidates["history_resolved_at"].notna() & (candidates["history_resolved_at"] <= candidates["sys_updated_at"]))

print("\nTotal candidate incidents:", len(candidates))
print(
    "Resolution after snapshot:",
    future_resolution.sum()
)
print(
    "Missing resolution timestamps:",
    missing_resolution.sum()
)
print(
    "Resolution at or before snapshot:",
    already_resolved.sum()
)

print(
    "\nTotal accounted candidates:", future_resolution.sum() + missing_resolution.sum() + already_resolved.sum()
)

# Examine potentially invalid prediction snapshots.
print("\nIncidents resolved at or before snapshot:")

print(
    candidates.loc[
        already_resolved,
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "history_resolved_at",
            "incident_state"
        ]
    ].head(5).to_string(index=False)
)

# Check how many candidate snapshots were recorded
# more than one hour after the ticket was opened.
print(
    "\nCandidate snapshots after 60 minutes:",
    (candidates["minutes_since_open"] > 60).sum()
)

#step 9: Investigate late incident snapshots

# Select candidate incidents whose earliest available update
late_snapshots = candidates[candidates["minutes_since_open"] > 60].copy()

print("\nTotal late snapshots:", len(late_snapshots))

# Convert elapsed minutes into hours.
late_snapshots["hours_since_open"] = (late_snapshots["minutes_since_open"] / 60)

print("\nMinimum hours since opening: ", late_snapshots["hours_since_open"].min())
print("Median hours since opening: ", late_snapshots["hours_since_open"].median())
print("Maximum hours since opening: ", late_snapshots["hours_since_open"].max())

# Examine incident states among late snapshots.
print("\nLate snapshot state distribution:")
print(late_snapshots["incident_state"].value_counts(dropna=False))

# Sort late snapshots from largest to smallest elapsed time.
late_snapshots = late_snapshots.sort_values(by="hours_since_open", ascending=False)

# Display the five incidents with the largest elapsed times.
print("\nFive latest incident snapshots:")

print(
    late_snapshots[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "incident_state",
            "hours_since_open"
        ]
    ].head(5).to_string(index=False)
)

#step 10: Establish early-snapshot eligibility

# Keep incidents whose earliest available snapshot occurred
early_candidates = candidates[candidates["minutes_since_open"] <= 60].copy()

print("\nTotal early candidates:", len(early_candidates))

# Count incidents resolved strictly after the input snapshot.
early_future_resolution = (early_candidates["history_resolved_at"] > early_candidates["sys_updated_at"])

# Count incidents without a known resolution timestamp.
early_missing_resolution = (early_candidates["history_resolved_at"].isna())

# Count incidents resolved at or before the input snapshot.
early_already_resolved = (early_candidates["history_resolved_at"].notna() & 
                          ( early_candidates["history_resolved_at"] <= early_candidates["sys_updated_at"]))

print("\nResolution after snapshot: ", early_future_resolution.sum())
print("Missing resolution timestamps: ", early_missing_resolution.sum())
print("Resolution at or before snapshot: ", early_already_resolved.sum())

# Verify that every early candidate is accounted for.
total_accounted = (
    early_future_resolution.sum() +
    early_missing_resolution.sum() +
    early_already_resolved.sum()
)

print("\nTotal accounted early candidates:", total_accounted)
print("Actual early candidate rows:", len(early_candidates))

#step 11: Investigate the original dataset columns

# history contains the original event-log columns.
original_columns = history.columns

print("\nOriginal dataset columns:")
print(original_columns.tolist())
print("\nNumber of original columns:", len(original_columns))

# Display only original columns, excluding the fields
# we created during our data-quality investigation.
print("\nSample early candidate records:")
print(
    early_candidates[original_columns]
    .head(3)
    .to_string(index=False)
)

# Inspect data types and missing values.
print("\nColumn data types and missing values:")

column_summary = pd.DataFrame({
    "dtype": early_candidates[original_columns].dtypes,
    "missing_count": early_candidates[original_columns].isna().sum()
})

print(column_summary.to_string())

#step 12:  Investigate missing and unknown feature values

feature_candidates = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom",
    "caller_id",
    "opened_by",
    "cmdb_ci",
    "assignment_group",
    "assigned_to"
]

# Create an empty list to store our findings.
feature_summary = []

for column in feature_candidates:
    # Count actual missing values (NaN).
    missing_count = early_candidates[column].isna().sum()
    
    # Count values represented by the string "?".
    unknown_count = (early_candidates[column] == "?").sum()
    
    # Count distinct values, including NaN if present.
    unique_count = early_candidates[column].nunique(dropna=False)
    
    # Store the results for this column.
    feature_summary.append({
        "column": column,
        "missing_count": missing_count,
        "unknown_count": unknown_count,
        "unique_count": unique_count
    })

# Convert our results into a summary DataFrame.
feature_summary = pd.DataFrame(feature_summary)

print("\nFeature missing-value summary:")
print(feature_summary.to_string(index=False))
    
# Step 13: Investigate feature stability across incident history

# Count distinct recorded values for each feature
# within the history of each incident.
feature_variations = (history.groupby("number")[feature_candidates].nunique(dropna=False))

# Restrict the investigation to our early candidate incidents.
feature_variations = feature_variations.reindex(early_candidates["number"])

# Create an empty list to store our findings.
stability_summary = []

for col in feature_candidates:

    # An incident has a recorded feature change if
    # more than one distinct value appears in its history.
    changed_count = (feature_variations[col] > 1).sum()

    stability_summary.append({
        "column": col,
        "incidents_with_changes": changed_count
    })

# Display the results as a summary table.
stability_summary = pd.DataFrame(stability_summary)

print("\nFeature stability across incident history:")
print(stability_summary.to_string(index=False))


# Step 14: Investigate priority labels and target leakage

# Examine the distribution of priority labels
# in our early candidate incidents.
print("\nPriority distribution:")

print(early_candidates["priority"].value_counts(dropna=False))

# Investigate changes in priority, impact, and urgency.
priority_columns = ["priority", "impact", "urgency"]

# Count distinct recorded values per incident.
priority_variations = (history.groupby("number")[priority_columns].nunique(dropna=False))

# Restrict the analysis to our early candidate incidents.
priority_variations = priority_variations.reindex(early_candidates["number"])

print("\nIncidents with recorded changes:")

for col in priority_columns:
    changed_count = (
        priority_variations[col] > 1
    ).sum()

    print(col, ":", changed_count)

# Investigate the relationship between impact, urgency, and priority in our early snapshots.
priority_mapping = (early_candidates.groupby(["impact", "urgency"])["priority"].nunique(dropna=False))

print("\nDistinct priority labels per impact-urgency combination:")
print(priority_mapping.to_string())

# Step 15: Compare early-snapshot priority with last recorded priority

# Convert historical update timestamps into datetime values.
history["sys_updated_at"] = pd.to_datetime(history["sys_updated_at"], dayfirst=True)

# Sort historical events and select the last recorded event for each incident.
last_events = (history.sort_values("sys_updated_at", kind="stable")
               .drop_duplicates(subset="number", keep="last")
               .set_index("number"))

# Create a separate DataFrame for target-label investigation.
priority_audit = early_candidates[["number", "sys_updated_at", "priority"]].copy()

# Match each incident with its last recorded priority.
priority_audit["last_priority"] = (priority_audit["number"].map(last_events["priority"]))

# Match each incident with its last recorded update timestamp.
priority_audit["last_updated_at"] = (priority_audit["number"].map(last_events["sys_updated_at"]))

# Identify incidents whose priority differs between the early snapshot and the last recorded event.
priority_changed = (priority_audit["priority"] != priority_audit["last_priority"])

print("\nIncidents with different early and last priority:",priority_changed.sum())
print("Incidents with unchanged early and last priority:",(~priority_changed).sum())
print("\nSample incidents with different priorities:")
print(
    priority_audit.loc[
        priority_changed,
        [
            "number",
            "sys_updated_at",
            "priority",
            "last_updated_at",
            "last_priority"
        ]
    ].head(5).to_string(index=False)
)


# Step 16: Investigate earlier historical resolutions

# Find the earliest recorded resolution timestamp for every incident in the complete event history.
first_resolution_history = (history.groupby("number")["resolved_at"].min())

# Create a separate DataFrame for our investigation.
resolution_audit = early_candidates[["number", "opened_at", "sys_updated_at", "history_resolved_at"]].copy()

# Map the earliest recorded resolution timestamp to the corresponding incident.
resolution_audit["first_resolved_at"] = (resolution_audit["number"].map(first_resolution_history))

# Select incidents whose latest resolution occurred after the input snapshot.
future_resolution_audit = (resolution_audit["history_resolved_at"] >resolution_audit["sys_updated_at"])

# Check whether an earlier resolution was already recorded at or before that same input snapshot.
earlier_resolution = (resolution_audit["first_resolved_at"].notna() &
    (
        resolution_audit["first_resolved_at"] <=
        resolution_audit["sys_updated_at"]
    )
)

# Incidents satisfying both conditions need investigation.
suspicious_incidents = resolution_audit[future_resolution_audit & earlier_resolution]

print("\nIncidents with an earlier resolution before the snapshot:",len(suspicious_incidents))
print("\nSample incidents:")
print(
    suspicious_incidents[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "first_resolved_at",
            "history_resolved_at"
        ]
    ].head(5).to_string(index=False)
)

#Step 17: Create separate eligible datasets
resolved_after_snapshot = (early_candidates["history_resolved_at"] > early_candidates["sys_updated_at"])

# Classification can also retain incidents without a resolution timestamp.
classification_data = early_candidates[resolved_after_snapshot | early_candidates["history_resolved_at"].isna()].copy()

# Regression requires a recorded resolution after the snapshot.
regression_data = early_candidates[resolved_after_snapshot].copy()

# Print dataset sizes.
print("\nClassification incidents:", len(classification_data))
print("Regression incidents:", len(regression_data))

# Check the priority labels available for classification.
print("\nClassification priority distribution:")
print(classification_data["priority"].value_counts())

# Verify every regression incident is included in classification.
regression_in_classification = (
    regression_data["number"]
    .isin(classification_data["number"])
    .all()
)

print("\nAll regression incidents included in classification:",regression_in_classification)

# STEP 18: Define initial model input features

classification_features = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom"
]

# Reuse classification features and add regression-specific inputs.
regression_features = classification_features + ["impact", "urgency"]

print("\nClassification features:")
print(classification_features)
print("Total:", len(classification_features))

print("\nRegression features:")
print(regression_features)
print("Total:", len(regression_features))

# STEP 19: Define the regression target

# Calculate remaining resolution time in hours.
regression_data["remaining_resolution_hours"] = (
    (
        regression_data["history_resolved_at"]
        - regression_data["sys_updated_at"]
    ).dt.total_seconds() / 3600
)

# Review the target distribution.
print( "\nMinimum remaining hours:", regression_data["remaining_resolution_hours"].min() )
print( "Maximum remaining hours:", regression_data["remaining_resolution_hours"].max() )
print( "Median remaining hours:", regression_data["remaining_resolution_hours"].median() )

# Check for invalid target values.
print( "Missing remaining hours:", regression_data["remaining_resolution_hours"].isna().sum() )
print( "Nonpositive remaining hours:", ( regression_data["remaining_resolution_hours"] <= 0 ).sum() )


# STEP 20: Investigate remaining resolution time distribution

# Calculate key percentiles.
print("\nRemaining resolution time percentiles (hours):")
print( regression_data["remaining_resolution_hours"].quantile( [0.50, 0.90, 0.95, 0.99] ) )

# Count incidents taking longer than 30 days.
over_30_days = (regression_data["remaining_resolution_hours"] > 720 ).sum()

# Count incidents taking longer than 90 days.
over_90_days = ( regression_data["remaining_resolution_hours"] > 2160 ).sum()

print("\nIncidents exceeding 30 days:", over_30_days)
print("Incidents exceeding 90 days:", over_90_days)

# Display the five longest-running incidents.
print("\nFive incidents with the longest remaining resolution times:")
print(
    regression_data.nlargest(
        5, "remaining_resolution_hours"
    )[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "history_resolved_at",
            "remaining_resolution_hours"
        ]
    ].to_string(index=False)
)
    
# STEP 21: Investigate feature quality

# Classification feature quality
print("\nClassification feature quality:")

for feature in classification_features:
    print(
        feature,
        "| Unknown '?':", classification_data[feature].eq("?").sum(),
        "| Missing:", classification_data[feature].isna().sum(),
        "| Unique:", classification_data[feature].nunique()
    )

# Regression feature quality
print("\nRegression feature quality:")

for feature in regression_features:
    print(
        feature,
        "| Unknown '?':", regression_data[feature].eq("?").sum(),
        "| Missing:", regression_data[feature].isna().sum(),
        "| Unique:", regression_data[feature].nunique()
    )
    
# STEP 22: Standardize missing values

# Create independent working copies.
classification_clean = classification_data.copy()
regression_clean = regression_data.copy()

# Replace '?' placeholders in classification features.
for feature in classification_features:
    classification_clean[feature] = classification_clean[feature].replace(to_replace="?", value=pd.NA)

# Replace '?' placeholders in regression features.
for feature in regression_features:
    regression_clean[feature] = regression_clean[feature].replace(to_replace="?", value=pd.NA)
    
# Verify missing values in classification features.
print("\nClassification missing values:")
print( classification_clean[classification_features] .isna() .sum() )

# Verify missing values in regression features.
print("\nRegression missing values:")
print( regression_clean[regression_features] .isna() .sum() )

# STEP 23: Create time-based features

time_features = [ "opened_hour", "opened_day_of_week" ]

# Classification: extract features from the opening timestamp.
classification_clean["opened_hour"] = classification_clean["opened_at"].dt.hour
classification_clean["opened_day_of_week"] = classification_clean["opened_at"].dt.day_of_week

# Regression: extract the same time-based features.
regression_clean["opened_hour"] = regression_clean["opened_at"].dt.hour
regression_clean["opened_day_of_week"] = regression_clean["opened_at"].dt.day_of_week

# Update the input feature lists.
classification_features = classification_features + time_features
regression_features = regression_features + time_features

# Verify the updated feature lists.
print("\nClassification features:", classification_features)
print("Total:", len(classification_features))

print("\nRegression features:", regression_features)
print("Total:", len(regression_features))

# Check whether the new features contain missing values.
print("\nClassification time feature missing values:")
print(classification_clean[time_features].isna().sum())

print("\nRegression time feature missing values:")
print(regression_clean[time_features].isna().sum())

# STEP 24: Create input features (X) and targets (y)

# Classification
X_classification = classification_clean[classification_features].copy()
y_classification = classification_clean["priority"].copy()

# Regression
X_regression = regression_clean[regression_features].copy()
y_regression = regression_clean["remaining_resolution_hours"].copy()

# Verify the dataset dimensions.
print("\nClassification:")
print("X shape:", X_classification.shape)
print("y shape:", y_classification.shape)

print("\nRegression:")
print("X shape:", X_regression.shape)
print("y shape:", y_regression.shape)

# Verify that neither target contains missing values.
print("\nMissing classification targets:", y_classification.isna().sum())
print("Missing regression targets:", y_regression.isna().sum())


# STEP 25: Validate model inputs and check for leakage

# Columns that must not be used to predict priority.
classification_forbidden = {
    "priority",
    "impact",
    "urgency",
    "resolution_hours",
    "remaining_resolution_hours",
    "history_resolved_at",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "closed_code"
}

# Columns that must not be used to predict resolution time.
regression_forbidden = {
    "remaining_resolution_hours",
    "resolution_hours",
    "history_resolved_at",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "closed_code"
}

# Verify that X contains exactly our selected features.
print( "\nClassification features match:", list(X_classification.columns) == classification_features )
print( "Regression features match:", list(X_regression.columns) == regression_features )

# Identify prohibited columns accidentally included in X.
print( "\nClassification prohibited columns:", sorted(set(X_classification.columns) & classification_forbidden) )
print( "Regression prohibited columns:", sorted(set(X_regression.columns) & regression_forbidden) )

# Verify that each input row matches its target row.
print( "\nClassification X/y indexes aligned:", X_classification.index.equals(y_classification.index) )
print( "Regression X/y indexes aligned:", X_regression.index.equals(y_regression.index) )

# STEP 26: Split classification and regression datasets
from sklearn.model_selection import train_test_split

X_class_train, X_class_test, y_class_train, y_class_test  = (
    train_test_split(
        X_classification,
        y_classification,
        test_size=0.20,
        random_state=42,
        stratify=y_classification
    )
)

# Regression: standard random split.
X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)

# Verify the resulting dimensions.
print("\nClassification split:")
print("X train:", X_class_train.shape)
print("X test:", X_class_test.shape)
print("y train:", y_class_train.shape)
print("y test:", y_class_test.shape)

print("\nRegression split:")
print("X train:", X_reg_train.shape)
print("X test:", X_reg_test.shape)
print("y train:", y_reg_train.shape)
print("y test:", y_reg_test.shape)

# Verify priority distribution after stratification.
print("\nClassification training priority distribution:")
print(y_class_train.value_counts())

print("\nClassification testing priority distribution:")
print(y_class_test.value_counts())

# STEP 27: Identify categorical and numerical features

# Classification
classification_categorical = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom"
]
classification_numerical = [ "opened_hour", "opened_day_of_week" ]

# Regression
regression_categorical = classification_categorical + [ "impact", "urgency" ]
regression_numerical = classification_numerical.copy()

# Print feature groups.
print("\nClassification categorical: ", classification_categorical)
print("Classification numerical: ", classification_numerical)
print("Regression categorical: ", regression_categorical)
print("Regression numerical: ", regression_numerical)

# Inspect actual training data types.
print("\nClassification training data types: \n", X_class_train.dtypes)
print("\nRegression training data types: \n", X_reg_train.dtypes)

# Inspect impact and urgency values.
print("\nImpact values: \n", X_reg_train["impact"].unique())
print("\nUrgency values: \n", X_reg_train["urgency"].unique())

# STEP 28: Define categorical preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #Replaces missing categorical values with "Unknown".
from sklearn.preprocessing import OneHotEncoder #Converts categorical values into numerical columns

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

print("\nCategorical preprocessing pipeline:")
print(categorical_pipeline)


# STEP 29: Define numerical preprocessing

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

print("\nNumerical preprocessing pipeline:")
print(numerical_pipeline)
    

# STEP 30: Combine categorical and numerical preprocessing
classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            classification_categorical
        ),
        (
            "numerical",
            numerical_pipeline,
            classification_numerical
        )
    ]
)

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            regression_categorical
        ),
        (
            "numerical",
            numerical_pipeline,
            regression_numerical
        )
    ]
)

print("\nClassification preprocessor:")
print(classification_preprocessor)

print("\nRegression preprocessor:")
print(regression_preprocessor)

# STEP 31: Fit preprocessing on training data only
X_class_train_processed = classification_preprocessor.fit_transform(X_class_train)
X_class_test_processed = classification_preprocessor.transform(X_class_test)

# Regression preprocessing
X_reg_train_processed = ( regression_preprocessor.fit_transform(X_reg_train) )
X_reg_test_processed = ( regression_preprocessor.transform(X_reg_test) )
    
# Verify processed dataset dimensions.
print("\nProcessed classification datasets:")
print("Training:", X_class_train_processed.shape)
print("Testing:", X_class_test_processed.shape)

print("\nProcessed regression datasets:")
print("Training:", X_reg_train_processed.shape)
print("Testing:", X_reg_test_processed.shape)

# STEP 32: Validate processed datasets

processed_datasets = {
    "Classification train": X_class_train_processed,
    "Classification test": X_class_test_processed,
    "Regression train": X_reg_train_processed,
    "Regression test": X_reg_test_processed
}

for name, dataset in processed_datasets.items():

    # Sparse matrices store their nonzero entries in .data.
    # Dense matrices store all values directly.
    values = dataset.data if issparse(dataset) else dataset

    print(
        name,
        "| All values finite:", np.isfinite(values).all(),
        "| Sparse:", issparse(dataset)
    )


# Verify that feature names match the processed column counts.
classification_feature_names = ( classification_preprocessor.get_feature_names_out() )
regression_feature_names = ( regression_preprocessor.get_feature_names_out() )

print( "\nClassification feature count matches: ", 
      len(classification_feature_names) == X_class_train_processed.shape[1] )
print( "Regression feature count matches:", len(regression_feature_names) == X_reg_train_processed.shape[1] )

# STEP 33: Train the Logistic Regression classifier

# Create the model.
logistic_model = LogisticRegression(max_iter=1000)

# Train the model using classification training data only.
logistic_model = logistic_model.fit(
    X_class_train_processed,
    y_class_train
)

# Inspect the trained model.
print("\nLogistic Regression training completed.")

print("\nLearned priority classes:")
print(logistic_model.classes_)

print("\nTraining iterations:")
print(logistic_model.n_iter_)

# STEP 34: Generate Logistic Regression predictions

# Predict priorities for the held-out test incidents.
y_class_pred = logistic_model.predict(X_class_test_processed)

# Verify the prediction count.
print("\nNumber of predictions:", len(y_class_pred))

# Inspect the first 10 predicted priorities.
print("\nFirst 10 predicted priorities:")
print(y_class_pred[:10])

# Inspect the distribution of predicted priority classes.
print("\nPredicted priority distribution:")
print( pd.Series(y_class_pred).value_counts() )

# STEP 35: Evaluate Logistic Regression

# Define the priority class order.
priority_labels = logistic_model.classes_

# 1. Overall accuracy
accuracy = accuracy_score(y_class_test, y_class_pred)
print("\nLogistic Regression accuracy:", accuracy)

# 2. Confusion matrix
cm = confusion_matrix(y_class_test, y_class_pred, labels=priority_labels)

# Convert the matrix into a labeled DataFrame.
cm_df = pd.DataFrame(
    cm,
    index=priority_labels,
    columns=priority_labels
)

print("\nConfusion matrix:")
print(cm_df.to_string())

# 3. Classification report
print("\nClassification report:")
print(
    classification_report(
        y_class_test,
        y_class_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# 4. Majority-class baseline accuracy
baseline_accuracy = (
    y_class_test.value_counts().max()
    / len(y_class_test)
)

print("\nMajority-class baseline accuracy:", baseline_accuracy)

# STEP 36: Train class-balanced Logistic Regression

# Create a separate model with balanced class weights.
balanced_logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# Fit the model using classification training data only.
balanced_logistic_model.fit(X_class_train_processed, y_class_train)

print("\nBalanced Logistic Regression training completed.")

# Verify the learned classes.
print("\nLearned priority classes:", balanced_logistic_model.classes_)

# Check training iterations.
print("\nTraining iterations:", balanced_logistic_model.n_iter_)

# Inspect predicted classes on TRAINING data only.
balanced_train_pred = balanced_logistic_model.predict( X_class_train_processed )
print("\nBalanced model training prediction distribution:")
print(pd.Series(balanced_train_pred).value_counts())


# STEP 37: Evaluate both models on training data

# Generate training predictions from the original model.
original_train_pred = logistic_model.predict( X_class_train_processed )

# Compare classification reports.
print("\nOriginal Logistic Regression - Training Report:")
print(
    classification_report(
        y_class_train,
        original_train_pred,
        labels=priority_labels,
        zero_division=0
    )
)
print("\nBalanced Logistic Regression - Training Report:")
print(
    classification_report(
        y_class_train,
        balanced_train_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# Generate the balanced model's confusion matrix.
balanced_cm = confusion_matrix( y_class_train, balanced_train_pred, labels=priority_labels )
balanced_cm_df = pd.DataFrame( balanced_cm, index=priority_labels, columns=priority_labels )

print("\nBalanced Model - Training Confusion Matrix:")
print(balanced_cm_df.to_string())

# STEP 38: Compare Logistic Regression models using cross-validation

# Create five stratified folds.
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42 )

# Original Logistic Regression configuration.
original_cv_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(classification_preprocessor)),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

# Class-balanced Logistic Regression configuration.
balanced_cv_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(classification_preprocessor)),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

# Evaluate both models using the original training features.
# Each fold fits its own preprocessing and model parameters.
original_cv_results = cross_validate(
    original_cv_pipeline,
    X_class_train,
    y_class_train,
    cv=cv,
    scoring=["f1_macro", "balanced_accuracy"],
    n_jobs=1
)

balanced_cv_results = cross_validate(
    balanced_cv_pipeline,
    X_class_train,
    y_class_train,
    cv=cv,
    scoring=["f1_macro", "balanced_accuracy"],
    n_jobs=1
)

# Print the cross-validation results.
print("\nOriginal Logistic Regression:")
print("Macro F1 per fold:", original_cv_results["test_f1_macro"])
print("Mean macro F1:", original_cv_results["test_f1_macro"].mean())
print(
    "Mean balanced accuracy:",
    original_cv_results["test_balanced_accuracy"].mean()
)

print("\nBalanced Logistic Regression:")
print("Macro F1 per fold:", balanced_cv_results["test_f1_macro"])
print("Mean macro F1:", balanced_cv_results["test_f1_macro"].mean())
print(
    "Mean balanced accuracy:",
    balanced_cv_results["test_balanced_accuracy"].mean()
)

# STEP 39: Train the Gaussian Naive Bayes classifier

# Create the model.
naive_bayes_model = GaussianNB()

# GaussianNB requires dense numerical input.
# Keep the original sparse training matrix unchanged.
X_class_train_dense = X_class_train_processed.toarray()

# Train using classification training data only.
naive_bayes_model.fit( X_class_train_dense, y_class_train )

print("\nGaussian Naive Bayes training completed.")

# Verify the learned priority classes.
print("\nLearned priority classes:")
print(naive_bayes_model.classes_)

# Check the training incident count for each class.
print("\nTraining incidents per class:")
print(naive_bayes_model.class_count_)

# STEP 40: Evaluate Gaussian Naive Bayes on training data

# Generate training predictions.
naive_bayes_train_pred = naive_bayes_model.predict( X_class_train_dense )

# Inspect the prediction distribution.
print("\nNumber of predictions:", len(naive_bayes_train_pred))

print("\nNaive Bayes training prediction distribution:")
print(pd.Series(naive_bayes_train_pred).value_counts())

# Calculate classification metrics.
print("\nNaive Bayes - Training Classification Report:")

print(
    classification_report(
        y_class_train,
        naive_bayes_train_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# Generate a labeled confusion matrix.
naive_bayes_cm = confusion_matrix( y_class_train, naive_bayes_train_pred, labels=priority_labels )
naive_bayes_cm_df = pd.DataFrame( naive_bayes_cm, index=priority_labels, columns=priority_labels )

print("\nNaive Bayes - Training Confusion Matrix:")
print(naive_bayes_cm_df.to_string())

# STEP 41: Cross-validation for Gaussian Naive Bayes

# Convert sparse matrices into dense arrays.
def sparse_to_dense(X):
    if issparse(X):
        return X.toarray()
    return X

# Create a complete preprocessing and training pipeline.
naive_bayes_cv_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(classification_preprocessor)
        ),
        (
            "to_dense",
            FunctionTransformer(
                sparse_to_dense,
                accept_sparse=True
            )
        ),
        (
            "model",
            GaussianNB()
        )
    ]
)

# Run cross-validation using the original training features.
naive_bayes_cv_results = cross_validate(
    naive_bayes_cv_pipeline,
    X_class_train,
    y_class_train,
    cv=cv,
    scoring=["f1_macro", "balanced_accuracy"],
    n_jobs=1
)

# Print validation results.
print("\nGaussian Naive Bayes - Cross-validation:")
print(
    "Macro F1 per fold:",
    naive_bayes_cv_results["test_f1_macro"]
)

print(
    "Mean macro F1:",
    naive_bayes_cv_results["test_f1_macro"].mean()
)

print(
    "Mean balanced accuracy:",
    naive_bayes_cv_results["test_balanced_accuracy"].mean()
)

# STEP 42: Train the linear SVM classifier

# Create the model.
svm_model = LinearSVC(
    max_iter=5000,
    random_state=42
)

# Train using classification training data only.
svm_model.fit( X_class_train_processed, y_class_train )

print("\nLinear SVM training completed.")

# Verify the learned priority classes.
print("\nLearned priority classes:")
print(svm_model.classes_)

# Inspect the number of training iterations.
print("\nTraining iterations:")
print(svm_model.n_iter_)


# STEP 43: Evaluate Linear SVM on training data

# Generate training predictions.
svm_train_pred = svm_model.predict(
    X_class_train_processed
)

# Inspect the prediction distribution.
print("\nNumber of predictions:", len(svm_train_pred))
print("\nLinear SVM training prediction distribution:")
print(pd.Series(svm_train_pred).value_counts())

# Calculate classification metrics.
print("\nLinear SVM - Training Classification Report:")
print(
    classification_report(
        y_class_train,
        svm_train_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# Generate a labeled confusion matrix.
svm_cm = confusion_matrix( y_class_train, svm_train_pred, labels=priority_labels )
svm_cm_df = pd.DataFrame( svm_cm, index=priority_labels, columns=priority_labels )

print("\nLinear SVM - Training Confusion Matrix:")
print(svm_cm_df.to_string())

# STEP 44: Cross-validation for Linear SVM

# Create a complete preprocessing and training pipeline.
svm_cv_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(classification_preprocessor)
        ),
        (
            "model",
            LinearSVC(
                max_iter=5000,
                random_state=42
            )
        )
    ]
)

# Evaluate SVM using five-fold stratified cross-validation.
svm_cv_results = cross_validate(
    svm_cv_pipeline,
    X_class_train,
    y_class_train,
    cv=cv,
    scoring=["f1_macro", "balanced_accuracy"],
    n_jobs=1
)

# Display validation results.
print("\nLinear SVM - Cross-validation:")

print(
    "Macro F1 per fold:",
    svm_cv_results["test_f1_macro"]
)

print(
    "Mean macro F1:",
    svm_cv_results["test_f1_macro"].mean()
)

print(
    "Mean balanced accuracy:",
    svm_cv_results["test_balanced_accuracy"].mean()
)


# STEP 45: Train class-balanced Linear SVM

# Create a separate model with balanced class weights.
balanced_svm_model = LinearSVC(
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

# Train using classification training data only.
balanced_svm_model.fit(
    X_class_train_processed,
    y_class_train
)

print("\nBalanced Linear SVM training completed.")

# Verify the learned priority classes.
print("\nLearned priority classes:")
print(balanced_svm_model.classes_)

# Inspect the number of training iterations.
print("\nTraining iterations:")
print(balanced_svm_model.n_iter_)

# Generate predictions using TRAINING data only.
balanced_svm_train_pred = balanced_svm_model.predict(
    X_class_train_processed
)

print("\nBalanced SVM training prediction distribution:")
print(pd.Series(balanced_svm_train_pred).value_counts())

# STEP 46: Evaluate class-balanced Linear SVM

# Calculate training classification metrics.
print("\nBalanced SVM - Training Classification Report:")

print(
    classification_report(
        y_class_train,
        balanced_svm_train_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# Generate the training confusion matrix.
balanced_svm_cm = confusion_matrix( y_class_train, balanced_svm_train_pred, labels=priority_labels )

# Add labels to make the matrix easier to interpret.
balanced_svm_cm_df = pd.DataFrame( balanced_svm_cm, index=priority_labels, columns=priority_labels )

print("\nBalanced SVM - Training Confusion Matrix:")
print(balanced_svm_cm_df.to_string())

# STEP 47: Cross-validation for balanced Linear SVM

# Create a complete pipeline with balanced class weights.
balanced_svm_cv_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(classification_preprocessor)
        ),
        (
            "model",
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            )
        )
    ]
)

# Run five-fold stratified cross-validation.
balanced_svm_cv_results = cross_validate(
    balanced_svm_cv_pipeline,
    X_class_train,
    y_class_train,
    cv=cv,
    scoring=["f1_macro", "balanced_accuracy"],
    n_jobs=1
)

# Display validation results.
print("\nBalanced Linear SVM - Cross-validation:")
print( "Macro F1 per fold:", balanced_svm_cv_results["test_f1_macro"] )
print( "Mean macro F1:", balanced_svm_cv_results["test_f1_macro"].mean() )
print( "Mean balanced accuracy:", balanced_svm_cv_results["test_balanced_accuracy"].mean() )

# STEP 48: Evaluate balanced SVM on held-out test data

# Generate predictions without fitting again.
balanced_svm_test_pred = balanced_svm_model.predict(
    X_class_test_processed
)

# Inspect the predicted priority distribution.
print("\nBalanced SVM test prediction distribution:")
print(pd.Series(balanced_svm_test_pred).value_counts())

# Calculate classification metrics.
print("\nBalanced SVM - Test Classification Report:")
print(
    classification_report(
        y_class_test,
        balanced_svm_test_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# Generate the test confusion matrix.
balanced_svm_test_cm = confusion_matrix( y_class_test, balanced_svm_test_pred, labels=priority_labels )
balanced_svm_test_cm_df = pd.DataFrame( balanced_svm_test_cm, index=priority_labels, columns=priority_labels )

print("\nBalanced SVM - Test Confusion Matrix:")
print(balanced_svm_test_cm_df.to_string())

# Calculate summary metrics.
print(
    "\nTest macro F1:",
    f1_score(
        y_class_test,
        balanced_svm_test_pred,
        average="macro"
    )
)

print(
    "Test balanced accuracy:",
    balanced_accuracy_score(
        y_class_test,
        balanced_svm_test_pred
    )
)
    

# Create the model.
linear_regression_model = LinearRegression()

# Train using existing regression training data.
linear_regression_model.fit( X_reg_train_processed, y_reg_train )

print("\nLinear Regression training completed.")
print("Number of features:", linear_regression_model.n_features_in_)

# STEP 50: Evaluate Linear Regression on training data

# Generate predictions using training data.
linear_regression_train_pred = linear_regression_model.predict( X_reg_train_processed )

# Calculate regression metrics.
train_mae = mean_absolute_error( y_reg_train, linear_regression_train_pred )
train_mse = mean_squared_error( y_reg_train, linear_regression_train_pred )
train_r2 = r2_score( y_reg_train, linear_regression_train_pred )

# Display results.
print("\nLinear Regression - Training Results:")
print("MAE (hours):", train_mae)
print("MSE (hours squared):", train_mse)
print("R²:", train_r2)

# STEP 51: Cross-validation for Linear Regression

# Create five shuffled folds.
regression_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Create a complete preprocessing and training pipeline.
linear_regression_cv_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(regression_preprocessor)),
        ("model", LinearRegression())
    ]
)

# Run cross-validation on original training features.
linear_regression_cv_results = cross_validate(
    linear_regression_cv_pipeline,
    X_reg_train,
    y_reg_train,
    cv=regression_cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2"
    ],
    n_jobs=1
)

# Display mean validation metrics.
print("\nLinear Regression - Cross-validation:")
print(
    "Mean validation MAE (hours):",
    -linear_regression_cv_results["test_neg_mean_absolute_error"].mean()
)

print(
    "Mean validation MSE (hours squared):",
    -linear_regression_cv_results["test_neg_mean_squared_error"].mean()
)

print(
    "Mean validation R²:",
    linear_regression_cv_results["test_r2"].mean()
)
      

# STEP 52: Evaluate Linear Regression on test data

# Generate predictions using the held-out test features.
linear_regression_test_pred = linear_regression_model.predict(
    X_reg_test_processed
)

# Calculate regression metrics.
test_mae = mean_absolute_error( y_reg_test, linear_regression_test_pred )
test_mse = mean_squared_error( y_reg_test, linear_regression_test_pred )
test_r2 = r2_score( y_reg_test, linear_regression_test_pred )

# Display results.
print("\nLinear Regression - Test Results:")
print("MAE (hours):", test_mae)
print("MSE (hours squared):", test_mse)
print("R²:", test_r2)
          
# STEP 53: Investigate Linear Regression prediction errors

# Separate incidents based on actual remaining resolution time.
short_incidents = y_reg_train <= 720
long_incidents = y_reg_train > 720

# Calculate MAE for incidents resolved within 30 days.
short_mae = mean_absolute_error( y_reg_train[short_incidents], linear_regression_train_pred[short_incidents] )

# Calculate MAE for incidents taking longer than 30 days.
long_mae = mean_absolute_error( y_reg_train[long_incidents], linear_regression_train_pred[long_incidents] )

print("\nLinear Regression - Error Analysis:")
print("\nIncidents resolved within 30 days:")
print("Count:", short_incidents.sum())
print("MAE (hours):", short_mae)

print("\nIncidents taking longer than 30 days:")
print("Count:", long_incidents.sum())
print("MAE (hours):", long_mae)

# Identify predictions that are physically impossible.
negative_predictions = ( linear_regression_train_pred < 0 ).sum()

print("\nNegative predictions:", negative_predictions)


# STEP 54: Establish a median regression baseline

# Create a model that always predicts the training median.
median_baseline_model = DummyRegressor(strategy="median")

# Train using existing regression training data.
median_baseline_model.fit( X_reg_train_processed, y_reg_train )

# Generate training predictions.
median_baseline_train_pred = median_baseline_model.predict( X_reg_train_processed )

print("\nMedian Baseline - Training Results:")
print("Constant prediction (hours):", median_baseline_model.constant_[0, 0])
print("MAE (hours):", mean_absolute_error(y_reg_train, median_baseline_train_pred))
print("MSE (hours squared):", mean_squared_error(y_reg_train, median_baseline_train_pred))
print("R²:", r2_score(y_reg_train, median_baseline_train_pred))


# STEP 55: Evaluate Median Baseline on test data

# Generate predictions using existing test features.
median_baseline_test_pred = median_baseline_model.predict( X_reg_test_processed )

# Calculate test metrics.
baseline_test_mae = mean_absolute_error( y_reg_test, median_baseline_test_pred )
baseline_test_mse = mean_squared_error( y_reg_test, median_baseline_test_pred )
baseline_test_r2 = r2_score( y_reg_test, median_baseline_test_pred )

# Display results.
print("\nMedian Baseline - Test Results:")
print("MAE (hours):", baseline_test_mae)
print("MSE (hours squared):", baseline_test_mse)
print("R²:", baseline_test_r2)

# STEP 56: Train Support Vector Regression

# Create the model.
svr_model = LinearSVR(
    max_iter=5000,
    random_state=42
)

# Train using existing regression training data.
svr_model.fit( X_reg_train_processed, y_reg_train )

print("\nLinear SVR training completed.")
print("Number of features:", svr_model.n_features_in_)

# STEP 57: Evaluate Linear SVR on training data

# Generate training predictions.
svr_train_pred = svr_model.predict( X_reg_train_processed )

# Calculate regression metrics.
svr_train_mae = mean_absolute_error( y_reg_train, svr_train_pred )
svr_train_mse = mean_squared_error( y_reg_train, svr_train_pred )
svr_train_r2 = r2_score( y_reg_train, svr_train_pred )

# Count negative predictions.
svr_negative_predictions = (svr_train_pred < 0).sum()

# Display results.
print("\nLinear SVR - Training Results:")
print("MAE (hours):", svr_train_mae)
print("MSE (hours squared):", svr_train_mse)
print("R²:", svr_train_r2)
print("Negative predictions:", svr_negative_predictions)


# STEP 58: Cross-validation for Linear SVR

# Create a complete preprocessing and training pipeline.
svr_cv_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(regression_preprocessor)),
        (
            "model",
            LinearSVR(
                max_iter=5000,
                random_state=42
            )
        )
    ]
)

# Run cross-validation on original training features.
svr_cv_results = cross_validate(
    svr_cv_pipeline,
    X_reg_train,
    y_reg_train,
    cv=regression_cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2"
    ],
    n_jobs=1
)

# Display mean validation metrics.
print("\nLinear SVR - Cross-validation:")
print( "Mean validation MAE (hours):", -svr_cv_results["test_neg_mean_absolute_error"].mean() )
print( "Mean validation MSE (hours squared):", -svr_cv_results["test_neg_mean_squared_error"].mean() )
print( "Mean validation R²:", svr_cv_results["test_r2"].mean() )

# STEP 59: Evaluate Linear SVR on test data

# Generate predictions using the test features.
svr_test_pred = svr_model.predict( X_reg_test_processed )

# Calculate regression metrics.
svr_test_mae = mean_absolute_error( y_reg_test, svr_test_pred )
svr_test_mse = mean_squared_error( y_reg_test, svr_test_pred )
svr_test_r2 = r2_score( y_reg_test, svr_test_pred )

# Count negative predictions.
svr_test_negative_predictions = ( svr_test_pred < 0 ).sum()

# Display results.
print("\nLinear SVR - Test Results:")
print("MAE (hours):", svr_test_mae)
print("MSE (hours squared):", svr_test_mse)
print("R²:", svr_test_r2)
print("Negative predictions:", svr_test_negative_predictions)

# STEP 60: Investigate Linear SVR prediction errors

# Calculate MAE for incidents resolved within 30 days.
svr_short_mae = mean_absolute_error( y_reg_train[short_incidents], svr_train_pred[short_incidents] )

# Calculate MAE for incidents taking longer than 30 days.
svr_long_mae = mean_absolute_error( y_reg_train[long_incidents], svr_train_pred[long_incidents] )

# Display results.
print("\nLinear SVR - Error Analysis:")

print("\nIncidents resolved within 30 days:")
print("Count:", short_incidents.sum())
print("MAE (hours):", svr_short_mae)

print("\nIncidents taking longer than 30 days:")
print("Count:", long_incidents.sum())
print("MAE (hours):", svr_long_mae)


# STEP 61: Investigate long-running incident predictions

# Calculate the mean actual remaining resolution time.
long_actual_mean = y_reg_train[long_incidents].mean()

# Calculate the mean predicted remaining resolution time.
long_predicted_mean = svr_train_pred[long_incidents].mean()

print("\nLinear SVR - Long-Running Incidents:")
print("Count:", long_incidents.sum())
print("Mean actual remaining hours:", long_actual_mean)
print("Mean predicted remaining hours:", long_predicted_mean)

# STEP 62: Measure underestimation of long-running incidents

# Extract actual and predicted durations.
long_actual = y_reg_train[long_incidents]
long_predicted = svr_train_pred[long_incidents]

# Identify incidents whose predicted duration is too low.
underestimated = long_predicted < long_actual.to_numpy()

# Count underestimated incidents.
underestimated_count = underestimated.sum()

# Calculate the percentage.
underestimated_percentage = (
    underestimated_count / len(long_actual)
) * 100

print("\nLinear SVR - Long-Running Incident Underestimation:")
print("Total long-running incidents:", len(long_actual))
print("Underestimated incidents:", underestimated_count)
print("Underestimated percentage:", underestimated_percentage)

# STEP 63: Train Linear Regression with a log-transformed target

# Create a Linear Regression model with target transformation.
log_linear_regression_model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1
)

# Train using existing regression training data.
log_linear_regression_model.fit( X_reg_train_processed, y_reg_train )

print("\nLog-Transformed Linear Regression training completed.")
print( "Number of features:", log_linear_regression_model.n_features_in_ )


# STEP 64: Evaluate log-transformed Linear Regression

# Generate predictions in original hours.
log_linear_regression_train_pred = ( log_linear_regression_model.predict( X_reg_train_processed ) )

# Calculate regression metrics.
log_train_mae = mean_absolute_error( y_reg_train, log_linear_regression_train_pred )
log_train_mse = mean_squared_error( y_reg_train, log_linear_regression_train_pred )
log_train_r2 = r2_score( y_reg_train, log_linear_regression_train_pred )

# Count negative predictions.
log_negative_predictions = ( log_linear_regression_train_pred < 0 ).sum()

# Display results.
print("\nLog-Transformed Linear Regression - Training Results:")
print("MAE (hours):", log_train_mae)
print("MSE (hours squared):", log_train_mse)
print("R²:", log_train_r2)
print("Negative predictions:", log_negative_predictions)


# STEP 65: Cross-validation for log-transformed Linear Regression

# Create a pipeline for feature preprocessing and regression.
log_linear_regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(regression_preprocessor)),
        ("model", LinearRegression())
    ]
)

# Wrap the pipeline to transform the target.
log_linear_regression_cv_model = TransformedTargetRegressor(
    regressor=log_linear_regression_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# Run five-fold cross-validation.
log_linear_regression_cv_results = cross_validate(
    log_linear_regression_cv_model,
    X_reg_train,
    y_reg_train,
    cv=regression_cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2"
    ],
    n_jobs=1
)

# Display the results.
print("\nLog-Transformed Linear Regression - Cross-validation:")
print( "Mean validation MAE (hours):", -log_linear_regression_cv_results["test_neg_mean_absolute_error"].mean() )
print( "Mean validation MSE (hours squared):", -log_linear_regression_cv_results["test_neg_mean_squared_error"].mean() )
print( "Mean validation R²:", log_linear_regression_cv_results["test_r2"].mean() )

# STEP 66: Examine individual cross-validation folds

print("\nLog-Transformed Linear Regression - Individual Folds:")

for fold in range(5):

    mae = -log_linear_regression_cv_results[
        "test_neg_mean_absolute_error"
    ][fold]

    mse = -log_linear_regression_cv_results[
        "test_neg_mean_squared_error"
    ][fold]

    r2 = log_linear_regression_cv_results["test_r2"][fold]

    print(f"\nFold {fold + 1}:")
    print("MAE (hours):", mae)
    print("MSE (hours squared):", mse)
    print("R²:", r2)


# STEP 67: Investigate Fold 3 prediction errors

# Recreate the same five cross-validation splits.
cv_splits = list(regression_cv.split(X_reg_train))

# Select Fold 3 (Python indexes start at zero).
fold3_train_idx, fold3_val_idx = cv_splits[2]

# Create a fresh copy of the existing CV model.
fold3_model = clone(log_linear_regression_cv_model)

# Train using Fold 3's training portion only.
fold3_model.fit( X_reg_train.iloc[fold3_train_idx], y_reg_train.iloc[fold3_train_idx] )

# Generate predictions for Fold 3's validation portion.
fold3_pred = fold3_model.predict( X_reg_train.iloc[fold3_val_idx] )

# Retrieve the corresponding actual resolution times.
fold3_actual = y_reg_train.iloc[fold3_val_idx].to_numpy()

print("\nFold 3 - Prediction Investigation:")

print("Maximum actual hours:", fold3_actual.max())
print("Maximum predicted hours:", fold3_pred.max())

# Calculate absolute prediction errors.
fold3_errors = np.abs(fold3_actual - fold3_pred)

# Find the incident with the largest absolute error.
worst_idx = np.argmax(fold3_errors)

print("\nIncident with the largest prediction error:")
print("Actual hours:", fold3_actual[worst_idx])
print("Predicted hours:", fold3_pred[worst_idx])
print("Absolute error (hours):", fold3_errors[worst_idx])

# STEP 68: Inspect the incident with the largest prediction error

# Find its position in the original regression training dataset.
worst_train_position = fold3_val_idx[worst_idx]

# Retrieve the corresponding original DataFrame index.
worst_incident_index = X_reg_train.index[worst_train_position]

# Retrieve the incident number.
worst_incident_number = regression_clean.loc[ worst_incident_index, "number" ]

print("\nIncident with the largest prediction error:")
print("Incident number:", worst_incident_number)

# Display its original regression input features.
print("\nOriginal regression features:")
print( X_reg_train.iloc[worst_train_position].to_string() )

# STEP 69: Investigate rare categorical features

# Retrieve Fold 3's original training features.
fold3_training_features = X_reg_train.iloc[fold3_train_idx]

# Retrieve the incident with the largest prediction error.
worst_incident_features = X_reg_train.iloc[ worst_train_position ]

print("\nINC0001853 - Categorical Feature Frequencies:")

# Count matching values in Fold 3's training data.
for feature in regression_categorical:

    feature_value = worst_incident_features[feature]

    matching_count = (
        fold3_training_features[feature] == feature_value
    ).sum()

    print(
        feature,
        "| Value:", feature_value,
        "| Training occurrences:", matching_count
    )

# STEP 70: Investigate the influence of Symptom 392

# Retrieve the original incident as a one-row DataFrame.
original_incident = X_reg_train.iloc[ [worst_train_position] ].copy()

# Create a separate copy for our experiment.
modified_incident = original_incident.copy()

# Replace only the symptom with a missing value.
modified_incident["u_symptom"] = pd.NA

# Generate a prediction for the modified incident.
modified_prediction = fold3_model.predict( modified_incident )[0]

# Compare the predictions.
print("\nINC0001853 - Symptom Investigation:")
print("Original symptom:", original_incident["u_symptom"].iloc[0])
print("Original prediction (hours):", fold3_pred[worst_idx])
print("Prediction with unknown symptom (hours):", modified_prediction)

# STEP 71: Investigate the training example for Symptom 392

# Find the incident containing Symptom 392.
symptom_392_mask = ( fold3_training_features["u_symptom"] == "Symptom 392" )

# Retrieve the matching training incident.
symptom_392_incident = fold3_training_features.loc[ symptom_392_mask ]

# Retrieve its original DataFrame index.
symptom_392_index = symptom_392_incident.index

# Retrieve its incident number.
symptom_392_number = regression_clean.loc[ symptom_392_index, "number" ]

# Retrieve its actual remaining resolution time.
symptom_392_actual = y_reg_train.loc[ symptom_392_index ]

print("\nSymptom 392 - Training Example:")
print("Incident number:")
print(symptom_392_number.to_string(index=False))
print("\nActual remaining resolution time (hours):")
print(symptom_392_actual.to_string(index=False))

# STEP 72: Train log-transformed Ridge Regression

# Create a Ridge model with a log-transformed target.
log_ridge_model = TransformedTargetRegressor(
    regressor=Ridge(alpha=1.0),
    func=np.log1p,
    inverse_func=np.expm1
)

# Train using existing processed training features.
log_ridge_model.fit( X_reg_train_processed, y_reg_train )

print("\nLog-Transformed Ridge Regression training completed.")
print( "Number of features:", log_ridge_model.n_features_in_ )

# STEP 73: Evaluate log-transformed Ridge Regression

# Generate training predictions in original hours.
log_ridge_train_pred = log_ridge_model.predict( X_reg_train_processed )

# Calculate regression metrics.
log_ridge_train_mae = mean_absolute_error( y_reg_train, log_ridge_train_pred )
log_ridge_train_mse = mean_squared_error( y_reg_train, log_ridge_train_pred )
log_ridge_train_r2 = r2_score( y_reg_train, log_ridge_train_pred )

# Count negative predictions.
log_ridge_negative_predictions = ( log_ridge_train_pred < 0 ).sum()

# Display results.
print("\nLog-Transformed Ridge Regression - Training Results:")
print("MAE (hours):", log_ridge_train_mae)
print("MSE (hours squared):", log_ridge_train_mse)
print("R²:", log_ridge_train_r2)
print("Negative predictions:", log_ridge_negative_predictions)
print("Maximum predicted hours:", log_ridge_train_pred.max())

# STEP 74: Cross-validation for log-transformed Ridge Regression

# Create a pipeline for feature preprocessing and Ridge Regression.
log_ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(regression_preprocessor)),
        ("model", Ridge(alpha=1.0))
    ]
)

# Apply the same target transformation as Step 72.
log_ridge_cv_model = TransformedTargetRegressor(
    regressor=log_ridge_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# Run five-fold cross-validation.
log_ridge_cv_results = cross_validate(
    log_ridge_cv_model,
    X_reg_train,
    y_reg_train,
    cv=regression_cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2"
    ],
    n_jobs=1
)

# Display mean validation metrics.
print("\nLog-Transformed Ridge Regression - Cross-validation:")
print( "Mean validation MAE (hours):", -log_ridge_cv_results["test_neg_mean_absolute_error"].mean() )
print( "Mean validation MSE (hours squared):", -log_ridge_cv_results["test_neg_mean_squared_error"].mean() )
print( "Mean validation R²:", log_ridge_cv_results["test_r2"].mean() )

# STEP 75: Examine individual Ridge Regression validation folds

print("\nLog-Transformed Ridge Regression - Individual Folds:")

for fold in range(5):

    mae = -log_ridge_cv_results[
        "test_neg_mean_absolute_error"
    ][fold]

    mse = -log_ridge_cv_results[
        "test_neg_mean_squared_error"
    ][fold]

    r2 = log_ridge_cv_results["test_r2"][fold]

    print(f"\nFold {fold + 1}:")
    print("MAE (hours):", mae)
    print("MSE (hours squared):", mse)
    print("R²:", r2)
    
# STEP 76: Investigate Ridge Regression's long-running incidents

# Calculate MAE for incidents resolved within 30 days.
ridge_short_mae = mean_absolute_error( y_reg_train[short_incidents], log_ridge_train_pred[short_incidents] )

# Calculate MAE for incidents taking longer than 30 days.
ridge_long_mae = mean_absolute_error( y_reg_train[long_incidents], log_ridge_train_pred[long_incidents] )

# Extract actual and predicted durations for long-running incidents.
ridge_long_actual = y_reg_train[long_incidents].to_numpy()
ridge_long_predicted = log_ridge_train_pred[long_incidents]

# Calculate mean predicted duration.
ridge_long_predicted_mean = ridge_long_predicted.mean()

# Count how many long-running incidents were underestimated.
ridge_underestimated_count = ( ridge_long_predicted < ridge_long_actual ).sum()
ridge_underestimated_percentage = ( ridge_underestimated_count / len(ridge_long_actual) ) * 100

# Display results.
print("\nLog-Transformed Ridge - Error Analysis:")
print("\nIncidents resolved within 30 days:")
print("Count:", short_incidents.sum())
print("MAE (hours):", ridge_short_mae)
print("\nIncidents taking longer than 30 days:")
print("Count:", long_incidents.sum())
print("MAE (hours):", ridge_long_mae)
print("\nLong-running incident predictions:")
print("Mean actual hours:", ridge_long_actual.mean())
print("Mean predicted hours:", ridge_long_predicted_mean)
print("Underestimated incidents:", ridge_underestimated_count)
print("Underestimated percentage:", ridge_underestimated_percentage)


# STEP 77: Consolidate regression cross-validation results

# Reuse the cross-validation results from previous steps.
regression_cv_models = {
    "Linear Regression": linear_regression_cv_results,
    "Linear SVR": svr_cv_results,
    "Log-Transformed Linear Regression": log_linear_regression_cv_results,
    "Log-Transformed Ridge": log_ridge_cv_results
}

# Collect mean validation metrics for each model.
regression_summary = []

for model_name, results in regression_cv_models.items():

    regression_summary.append({
        "Model": model_name,

        "CV MAE (hours)":
            -results["test_neg_mean_absolute_error"].mean(),

        "CV MSE (hours squared)":
            -results["test_neg_mean_squared_error"].mean(),

        "CV R²":
            results["test_r2"].mean()
    })

# Convert results into a DataFrame.
regression_summary_df = pd.DataFrame(regression_summary)

print("\nResolveAI - Regression Cross-Validation Summary:")
print( regression_summary_df.round(4).to_string(index=False) )

# STEP 78: Consolidate classification cross-validation results

# Reuse existing cross-validation results.
classification_cv_models = {
    "Original Logistic Regression": original_cv_results,
    "Balanced Logistic Regression": balanced_cv_results,
    "Gaussian Naive Bayes": naive_bayes_cv_results,
    "Original Linear SVM": svm_cv_results,
    "Balanced Linear SVM": balanced_svm_cv_results
}

# Collect the mean validation metrics.
classification_summary = []

for model_name, results in classification_cv_models.items():

    classification_summary.append({
        "Model": model_name,

        "CV Macro F1":
            results["test_f1_macro"].mean(),

        "CV Balanced Accuracy":
            results["test_balanced_accuracy"].mean()
    })

# Convert the results into a DataFrame.
classification_summary_df = pd.DataFrame(
    classification_summary
)

print("\nResolveAI - Classification Cross-Validation Summary:")
print( classification_summary_df.round(4).to_string(index=False) )
      
      
clf_model = SVC(kernel="linear", C=1)
clf_model = clf_model.fit(X_class_train_processed, y_class_train)
print("CLF SVC training accuracy: ", clf_model.score(X_class_train_processed, y_class_train))


      
      
      
      
      
      
      
      
      
      
      
      

      
      
      
      

    
       
    
