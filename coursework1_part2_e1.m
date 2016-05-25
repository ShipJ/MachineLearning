%% Mathematical Programming and Research Methods Coursework 1

%% Part 2 - Linear Regression

%% Exercise 1(a)

% Create the given data set
data_x = [1 2 3 4]';
data_y = [3 2 0 5]';

% Create each polynomial basis of dimension k = 1,2,3,4
A_basis_1 = [ones(size(data_x))];
A_basis_2 = [ones(size(data_x)) data_x];
A_basis_3 = [ones(size(data_x)) data_x data_x.^2];
A_basis_4 = [ones(size(data_x)) data_x data_x.^2 data_x.^3];;

% Compute the coefficients of the elements of bases given above
% These correspond to the coefficients of the polynomial curve fitted using each individual basis 
c_basis_1 = (A_basis_1'*A_basis_1)\(A_basis_1'*data_y);
c_basis_2 = (A_basis_2'*A_basis_2)\(A_basis_2'*data_y);
c_basis_3 = (A_basis_3'*A_basis_3)\(A_basis_3'*data_y);
c_basis_4 = (A_basis_4'*A_basis_4)\(A_basis_4'*data_y);

% The fitted 'curve' for k = 1 is constant over all points
x_fit_1 = data_x;

% Compute the equations for each fitted curve over the defined range
x_fit_2 = linspace(0,10);
y_fit_1 = c_basis_1(1);
y_fit_2 = c_basis_2(2)*x_fit_2 + c_basis_2(1);
y_fit_3 = c_basis_3(3)*x_fit_2.^2 + c_basis_3(2)*x_fit_2 + c_basis_3(1);
y_fit_4 = c_basis_4(4)*x_fit_2.^3 + c_basis_4(3)*x_fit_2.^2 + c_basis_4(2)*x_fit_2 + c_basis_4(1);

% Compute the y-values corresponding to the curve at each data point
corresponding_y_1 = c_basis_2(1);
corresponding_y_2 = c_basis_2(2)*data_x + c_basis_2(1);
corresponding_y_3 = c_basis_3(3)*data_x.^2 + c_basis_3(2)*data_x + c_basis_3(1);
corresponding_y_4 = c_basis_4(4)*data_x.^3 + c_basis_4(3)*data_x.^2 + c_basis_4(2)*data_x + c_basis_4(1);

%% Plot of results
figure
hold on
% Plot the original data points
scatter(data_x,data_y,'filled','MarkerEdgeColor','black');
% Plot each fitted curve with polynomial basis of size k = 4,3,2,1
plot(x_fit_2,y_fit_4,x_fit_2,y_fit_3,x_fit_2,y_fit_2,'LineWidth',1.5);grid on;   
% Plot the curve for the k = 1 case separately as it is constant, only want it to appear atop each data point
plot(x_fit_1,y_fit_1,'-X','Color','black','MarkerSize',10,'LineWidth',1.5);legend('Initial Data Points','Basis of dim: k = 4', 'Basis of dim: k = 3','Basis of dim: k = 2','Basis of dim: k = 1','Location','southeast');title('Plot of each fitted curve corresponding to a chosen basis','FontSize',16);
axis([0 5 -5 10]);
set(gcf,'color','w');
hold off


%% Exercise 1(b)

% Equations given in answer sheet.


%% Exercise 1(c)

% Compute the residual errors between the fitted curve and the data set
errors_1 = data_y - corresponding_y_1;
errors_2 = data_y - corresponding_y_2;
errors_3 = data_y - corresponding_y_3;
errors_4 = data_y - corresponding_y_4;

% Compute the mean of the squared residual errors
mse_1 = mean(errors_1.^2)
mse_2 = mean(errors_2.^2)
mse_3 = mean(errors_3.^2)
mse_4 = mean(errors_4.^2)
















