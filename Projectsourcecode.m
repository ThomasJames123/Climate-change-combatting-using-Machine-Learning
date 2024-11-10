% Loading file after importation in Matlab
data = readtable('GlobalLandTemperaturesByCity');
% Calculating the average temperatures for each city.
city_avg_temps = varfun(@mean, GlobalLandTemperaturesByCity, 'InputVariables', 'AverageTemperature', 'GroupingVariables', 'City');
% Displaying the average temperatures for each city.
disp(city_avg_temps);
% Calculating the baseline average temperature for each city
if isnumeric(GlobalLandTemperaturesByCity.Year) GlobalLandTemperaturesByCity.Year = datetime(GlobalLandTemperaturesByCity.Year, 1, 1);end 
ref_period_start = datetime(1951, 1, 1);
ref_period_end = datetime(1980, 12, 31);
ref_data = GlobalLandTemperaturesByCity(GlobalLandTemperaturesByCity.Year >= ref_period_start & GlobalLandTemperaturesByCity.Year <= ref_period_end, :);
baseline_avg_temps = varfun(@mean, ref_data, 'InputVariables', 'AverageTemperature', 'GroupingVariables', 'City');
baseline_avg_temps.Properties.VariableNames{'GroupCount'} = 'BaselineAvgTemperature';
GlobalLandTemperaturesByCity = outerjoin(GlobalLandTemperaturesByCity, baseline_avg_temps, 'Keys', 'City', 'MergeKeys',true);
GlobalLandTemperaturesByCity.TemperatureAnomaly=GlobalLandTemperaturesByCity.AverageTemperature - GlobalLandTemperaturesByCity.BaselineAvgTemperature;
% Calculating the temperature anomalies for each city
GlobalLandTemperaturesByCity.TemperatureAnomaly=GlobalLandTemperaturesByCity.AverageTemperature -GlobalLandTemperaturesByCity.BaselineAvgTemperature;
% Calculating the temperature anomalies for Abakan city in Russia
city_name = 'Abakan';
city_data = GlobalLandTemperaturesByCity(strcmp(GlobalLandTemperaturesByCity.City, city_name), :);
% Plotting the temperature anomalies for Abakan city in Russia
figure; 
plot(city_data.Year, city_data.TemperatureAnomaly, '-o'); 
xlabel('Year'); 
ylabel('Temperature Anomaly (°C)'); 
title(['Temperature Anomalies in ', city_name]); 
grid on;
% Calculating the mean average temperature anomalies for each country
country_avg_temps = varfun(@mean, GlobalLandTemperaturesByCity, 'InputVariables', 'AverageTemperature', 'GroupingVariables', 'Country');
% Plot of average temperatures anomalies over time
plot(data.Year,data.AverageTemperature);title('Global Land Temperatures By city over time');xlabel('Year');ylabel('Average Temperature');
% Predictive modelling using Polynomial regression
num_years_future = 50;
last_year = max(data.Year);
future_dates = (year(last_year) + 1):(year(last_year) + num_years_future);
future_dates = datetime(future_dates, 1, 1);
p = polyfit(year(data.Year), data.AverageTemperature, 1);
future_temps = polyval(p, year(future_dates));
% Plotting prediction made using Polynomial regression
figure; plot(data.Year, data.AverageTemperature, 'b'); hold on;
plot(future_dates, future_temps, 'r--');
title('Predicted Global Land Average Temperatures'); xlabel('Year'); ylabel('Temperature'); legend('Past Global Land Temperatures', 'Predicted Global Land Temperatures');
grid on;
hold off;
% Predictive modelling using Artificial Neural Network
%splitting the data into training and testing sets
splitRatio = 0.8;
splitIndex = round(splitRatio * length(data.Year));

trainDates = data.Year;
trainTemps = data.AverageTemperature;
net = fitnet([10, 10]);

trainDates = trainDates(:)';
trainTemps = trainTemps(:)';
net = fitnet(10);

trainDates = num2cell(trainDates);
trainTemps = num2cell(trainTemps);
net = fitnet(10);

trainDates = year(data.Year);
trainTemps = data.AverageTemperature;
net = fitnet(10);
[net, tr] = train(net, trainDates, trainTemps);

sample_ratio = 0.1;
sample_indices = randperm(length(trainDates), round(sample_ratio * length(trainDates)));
trainDates_sampled = trainDates(sample_indices);
trainTemps_sampled = trainTemps(sample_indices);
[net, tr] = train(net, trainDates_sampled, trainTemps_sampled); %#ok<*ASGLU>

num_years_future =100;
last_year = max(data.Year);
future_dates = (year(last_year) + 1):(year(last_year) + num_years_future);
future_dates = datetime(future_dates, 1, 1); %#ok<*NASGU>
inputSize = net.inputs{1}.size; disp(['Expected Input Size: ', num2str(inputSize)]);

last_year = 2013;
num_years_future = 100;
future_dates_numeric = (last_year + 1):(last_year + num_years_future);
future_dates_numeric = reshape(future_dates_numeric, [1, length(future_dates_numeric)]);
years = data.Year;
averageTemperatures = data.AverageTemperature;
numTrain = round(0.8 * length(years));
trainYears = years(1:numTrain);
trainTemps = averageTemperatures(1:numTrain);
testYears = years(numTrain+1:end);
testTemps = averageTemperatures(numTrain+1:end);
if isa(trainYears, 'datetime') trainYears = year(trainYears); end
if isa(testYears, 'datetime') testYears = year(testYears); end %#ok<*SEPEX>
trainYears = trainYears(:)';
trainTemps = trainTemps(:)';
testYears = testYears(:)';
testTemps = testTemps(:)';
net = fitnet(10);
[net, tr] = train(net, trainYears, trainTemps);
disp(tr);
predictedTempsTrain = net(trainYears);
predictedTempsTest = net(testYears);

last_year = 2013;
last_year = max(trainYears);
num_years_future = 100;
future_dates = (last_year + 1):(last_year + num_years_future);
future_dates_numeric = reshape(future_dates, 1, []);
predicted_temps = net(future_dates_numeric);

% Plotting prediction of average temperatures made using Artificial Neural Networks
figure;
plot(trainYears, trainTemps, 'b', 'DisplayName', 'Actual Training Data');
hold on;
plot(trainYears, predictedTempsTrain, 'r--', 'DisplayName', 'Predicted Training Data');
plot(testYears, testTemps, 'g', 'DisplayName', 'Actual Testing Data');
plot(testYears, predictedTempsTest, 'm--', 'DisplayName', 'Predicted Testing Data');
title('Average Temperature Prediction using ANN');
xlabel('Years');
ylabel('Average Temperatures');
legend show;
grid on;

figure;
plot(trainYears, trainTemps, 'b', 'DisplayName', 'Actual Training Data');
hold on;
plot(future_dates, predicted_temps, 'r--', 'DisplayName', 'Predicted Future Data using ANN');
title('Predicted Average Temperatures for the Next 100 Years');
xlabel('Years');
ylabel('Predicted Average Temperatures');
legend show;
grid on;
% Evaluating Artificial Neural Network model using MSE metrics
mseValue = mean((predicted_temps - testTemps).^2);
Arrays have incompatible sizes for this operation.
Related documentation
if length(predicted_temps) ~= length(testTemps) %#ok<ALIGN>
    disp('Arrays have incompatible sizes for MSE calculation.');
minLength = min(length(predicted_temps), length(testTemps)); predicted_temps = predicted_temps(1:minLength); testTemps = testTemps(1:minLength); end
Arrays have incompatible sizes for MSE calculation.
mseValue = mean((predicted_temps - testTemps).^2);
disp(['MSE Value: ', num2str(mseValue)]);
%Scenario Analysis
last_year = 2013;
num_years_future = 30;
last_year = max(data.Year);
future_dates = (year(last_year) + 1):(year(last_year) + num_years_future);
future_dates = datetime(future_dates, 1, 1);
p_bau = polyfit(year(data.Year), data.AverageTemperature, 1);
future_temps_bau = polyval(p_bau, year(future_dates));
p_mod = polyfit(year(data.Year), data.AverageTemperature, 1) * 0.75;
future_temps_mod = polyval(p_mod, year(future_dates));
p_agg = polyfit(year(data.Year), data.AverageTemperature, 1) * 0.5;
future_temps_agg = polyval(p_agg, year(future_dates));
% Plotting results of Scenario Analysis
figure;
plot(data.Year, data.AverageTemperature, 'b', 'DisplayName', 'Historical Data');
hold on;
plot(future_dates, future_temps_bau, 'r--', 'DisplayName', 'Predicted data using BAU Scenario');
plot(future_dates, future_temps_mod, 'g--', 'DisplayName', 'Moderate Mitigation');
plot(future_dates, future_temps_agg, 'm--', 'DisplayName', 'Aggressive Mitigation');
title('Global Land Temperature using Scenario Analysis');
xlabel('Years');
ylabel('Global Average Land Temperatures');
legend('show');
grid on;
hold off;
% Optimization using Least Squares 
years = data.Year;
averageTemperatures = data.AverageTemperature;
if isa(years, 'datetime') years = year(years);end
years = years(:);
averageTemperatures = averageTemperatures(:);
X = [years, ones(length(years), 1)];
coefficients = X \ averageTemperatures;
m = coefficients(1);
b = coefficients(2);
% To display slope
disp(['Slope (m): ', num2str(m)]);
% To display intercept
disp(['Intercept (b): ', num2str(b)]);
% Evaluation of Least Squares optimization model 
% Calculate the Mean Squared Error (MSE)
mseValue = mean((fittedTemps - averageTemperatures).^2);
disp(['Mean Squared Error (MSE): ', num2str(mseValue)]);
Mean Squared Error (MSE): 103.5441



