%% Mathematical Programming and Research Methods Coursework 1

%% Part 2 - Linear Regression

%% Exercise 2(d)

x_samples = rand(1000,1);
x_samples_num = numel(x_samples);
data_points = [];

% Generate corresponding y-values using random function 'gsig'
for i = 1:x_samples_num
    data_points = [data_points gsig(0.07,x_samples(i))];
end

% Basis of size 2
A_basis_2 = [ones(size(x_samples)) x_samples];
c_basis_2 = (A_basis_2'*A_basis_2)\(A_basis_2'*data_points');
y_fit_2 = c_basis_2(2)*x_samples + c_basis_2(1);

% Basis of size 18
A_basis_18 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13 x_samples.^14 x_samples.^15 x_samples.^16 x_samples.^17];
c_basis_18 = (A_basis_18'*A_basis_18)\(A_basis_18'*data_points');
y_fit_18 = c_basis_18(18)*x_samples.^17 + c_basis_18(17)*x_samples.^16 + c_basis_18(16)*x_samples.^15 + c_basis_18(15)*x_samples.^14 + c_basis_18(14)*x_samples.^13 + c_basis_18(13)*x_samples.^12 + c_basis_18(12)*x_samples.^11 + c_basis_18(11)*x_samples.^10 + c_basis_18(10)*x_samples.^9 + c_basis_18(9)*x_samples.^8 + c_basis_18(8)*x_samples.^7 + c_basis_18(7)*x_samples.^6 + c_basis_18(6)*x_samples.^5 + c_basis_18(5)*x_samples.^4 + c_basis_18(4)*x_samples.^3 + c_basis_18(3)*x_samples.^2 + c_basis_18(2)*x_samples + c_basis_18(1);

% Basis of size 5
A_basis_5 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4];
c_basis_5 = (A_basis_5'*A_basis_5)\(A_basis_5'*data_points');
y_fit_5 = c_basis_5(5)*x_samples.^4 + c_basis_5(4)*x_samples.^3 + c_basis_5(3)*x_samples.^2 + c_basis_5(2)*x_samples + c_basis_5(1);

%% Plot results
figure
hold on;
% Scatter plot of data points
scatter(x_samples,data_points,20,'filled'); grid on;
% Fitted polynomial curve with basis k = 2
plot(x_samples,y_fit_2);grid on;legend('Data Points','k = 2');
hold off

figure 
plot(x_samples,y_fit_5);grid on;
figure
scatter(x_samples,y_fit_18);grid on;






