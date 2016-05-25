%% Mathematical Programming and Research Methods Coursework 1

%% Part 2 - Linear Regression

%% Exercise 2(e)

% Generate 30 random x-values and order them
x_samples = sort(rand(1000,1));

% True plot of sin(2*pi*x) over interval [0,1]
x_vals = linspace(min(x_samples),max(x_samples));
y_vals = sin(2*pi*x_vals).*sin(2*pi*x_vals);

% Calculate Corresponding y-values
data_points = [];
data_points_num = numel(x_samples);

for i = 1:data_points_num
    data_points = [data_points gsig(0.07,x_samples(i))];
end


%% Exercise 2(bii)
% Plot over polynomial bases with sizes 2,5,10,14,18

% % Basis of size 1
% A_basis_1 = [ones(size(x_samples))];
% c_basis_1 = (A_basis_1'*A_basis_1)\(A_basis_1'*data_points');
% y_fit_1 = c_basis_1(1);

 % Basis of size 2
A_basis_2 = [ones(size(x_samples)) x_samples];
c_basis_2 = (A_basis_2'*A_basis_2)\(A_basis_2'*data_points');
y_fit_2 = c_basis_2(2)*x_vals + c_basis_2(1);

% Basis of size 3
A_basis_3 = [ones(size(x_samples)) x_samples x_samples.^2];
c_basis_3 = (A_basis_3'*A_basis_3)\(A_basis_3'*data_points');
y_fit_3 = c_basis_3(3)*x_vals.^2 + c_basis_3(2)*x_vals + c_basis_3(1);

% Basis of size 4
A_basis_4 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3];
c_basis_4 = (A_basis_4'*A_basis_4)\(A_basis_4'*data_points');
y_fit_4 = c_basis_4(4)*x_vals.^3 + c_basis_4(3)*x_vals.^2 + c_basis_4(2)*x_vals + c_basis_4(1);

% Basis of size 5
A_basis_5 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4];
c_basis_5 = (A_basis_5'*A_basis_5)\(A_basis_5'*data_points');
y_fit_5 = c_basis_5(5)*x_vals.^4 + c_basis_5(4)*x_vals.^3 + c_basis_5(3)*x_vals.^2 + c_basis_5(2)*x_vals + c_basis_5(1);

% Basis of size 6
A_basis_6 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5];
c_basis_6 = (A_basis_6'*A_basis_6)\(A_basis_6'*data_points');
y_fit_6 = c_basis_6(6)*x_vals.^5 + c_basis_6(5)*x_vals.^4 + c_basis_6(4)*x_vals.^3 + c_basis_6(3)*x_vals.^2 + c_basis_6(2)*x_vals + c_basis_6(1);

% Basis of size 7
A_basis_7 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6];
c_basis_7 = (A_basis_7'*A_basis_7)\(A_basis_7'*data_points');
y_fit_7 = c_basis_7(7)*x_vals.^6 + c_basis_7(6)*x_vals.^5 + c_basis_7(5)*x_vals.^4 + c_basis_7(4)*x_vals.^3 + c_basis_7(3)*x_vals.^2 + c_basis_7(2)*x_vals + c_basis_7(1);

% Basis of size 8
A_basis_8 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7];
c_basis_8 = (A_basis_8'*A_basis_8)\(A_basis_8'*data_points');
y_fit_8 = c_basis_8(8)*x_vals.^7 + c_basis_8(7)*x_vals.^6 + c_basis_8(6)*x_vals.^5 + c_basis_8(5)*x_vals.^4 + c_basis_8(4)*x_vals.^3 + c_basis_8(3)*x_vals.^2 + c_basis_8(2)*x_vals + c_basis_8(1);

% Basis of size 9
A_basis_9 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8];
c_basis_9 = (A_basis_9'*A_basis_9)\(A_basis_9'*data_points');
y_fit_9 = c_basis_9(9)*x_vals.^8 + c_basis_9(8)*x_vals.^7 + c_basis_9(7)*x_vals.^6 + c_basis_9(6)*x_vals.^5 + c_basis_9(5)*x_vals.^4 + c_basis_9(4)*x_vals.^3 + c_basis_9(3)*x_vals.^2 + c_basis_9(2)*x_vals + c_basis_9(1);

% Basis of size 10
A_basis_10 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9];
c_basis_10 = (A_basis_10'*A_basis_10)\(A_basis_10'*data_points');
y_fit_10 = c_basis_10(10)*x_vals.^9 + c_basis_10(9)*x_vals.^8 + c_basis_10(8)*x_vals.^7 + c_basis_10(7)*x_vals.^6 + c_basis_10(6)*x_vals.^5 + c_basis_10(5)*x_vals.^4 + c_basis_10(4)*x_vals.^3 + c_basis_10(3)*x_vals.^2 + c_basis_10(2)*x_vals + c_basis_10(1);

% Basis of size 11
A_basis_11 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10];
c_basis_11 = (A_basis_11'*A_basis_11)\(A_basis_11'*data_points');
y_fit_11 = c_basis_11(11)*x_vals.^10 + c_basis_11(10)*x_vals.^9 + c_basis_11(9)*x_vals.^8 + c_basis_11(8)*x_vals.^7 + c_basis_11(7)*x_vals.^6 + c_basis_11(6)*x_vals.^5 + c_basis_11(5)*x_vals.^4 + c_basis_11(4)*x_vals.^3 + c_basis_11(3)*x_vals.^2 + c_basis_11(2)*x_vals + c_basis_11(1);

% Basis of size 12
A_basis_12 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11];
c_basis_12 = (A_basis_12'*A_basis_12)\(A_basis_12'*data_points');
y_fit_12 = c_basis_12(12)*x_vals.^11 + c_basis_12(11)*x_vals.^10 + c_basis_12(10)*x_vals.^9 + c_basis_12(9)*x_vals.^8 + c_basis_12(8)*x_vals.^7 + c_basis_12(7)*x_vals.^6 + c_basis_12(6)*x_vals.^5 + c_basis_12(5)*x_vals.^4 + c_basis_12(4)*x_vals.^3 + c_basis_12(3)*x_vals.^2 + c_basis_12(2)*x_vals + c_basis_12(1);

% Basis of size 13
A_basis_13 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12];
c_basis_13 = (A_basis_13'*A_basis_13)\(A_basis_13'*data_points');
y_fit_13 = c_basis_13(13)*x_vals.^12 + c_basis_13(12)*x_vals.^11 + c_basis_13(11)*x_vals.^10 + c_basis_13(10)*x_vals.^9 + c_basis_13(9)*x_vals.^8 + c_basis_13(8)*x_vals.^7 + c_basis_13(7)*x_vals.^6 + c_basis_13(6)*x_vals.^5 + c_basis_13(5)*x_vals.^4 + c_basis_13(4)*x_vals.^3 + c_basis_13(3)*x_vals.^2 + c_basis_13(2)*x_vals + c_basis_13(1);

% Basis of size 14
A_basis_14 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13];
c_basis_14 = (A_basis_14'*A_basis_14)\(A_basis_14'*data_points');
y_fit_14 = c_basis_14(14)*x_vals.^13 + c_basis_14(13)*x_vals.^12 + c_basis_14(12)*x_vals.^11 + c_basis_14(11)*x_vals.^10 + c_basis_14(10)*x_vals.^9 + c_basis_14(9)*x_vals.^8 + c_basis_14(8)*x_vals.^7 + c_basis_14(7)*x_vals.^6 + c_basis_14(6)*x_vals.^5 + c_basis_14(5)*x_vals.^4 + c_basis_14(4)*x_vals.^3 + c_basis_14(3)*x_vals.^2 + c_basis_14(2)*x_vals + c_basis_14(1);

% Basis of size 15
A_basis_15 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13 x_samples.^14];
c_basis_15 = (A_basis_15'*A_basis_15)\(A_basis_15'*data_points');
y_fit_15 = c_basis_15(15)*x_vals.^14 + c_basis_15(14)*x_vals.^13 + c_basis_15(13)*x_vals.^12 + c_basis_15(12)*x_vals.^11 + c_basis_15(11)*x_vals.^10 + c_basis_15(10)*x_vals.^9 + c_basis_15(9)*x_vals.^8 + c_basis_15(8)*x_vals.^7 + c_basis_15(7)*x_vals.^6 + c_basis_15(6)*x_vals.^5 + c_basis_15(5)*x_vals.^4 + c_basis_15(4)*x_vals.^3 + c_basis_15(3)*x_vals.^2 + c_basis_15(2)*x_vals + c_basis_15(1);

% Basis of size 16
A_basis_16 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13 x_samples.^14 x_samples.^15];
c_basis_16 = (A_basis_16'*A_basis_16)\(A_basis_16'*data_points');
y_fit_16 = c_basis_16(16)*x_vals.^15 + c_basis_16(15)*x_vals.^14 + c_basis_16(14)*x_vals.^13 + c_basis_16(13)*x_vals.^12 + c_basis_16(12)*x_vals.^11 + c_basis_16(11)*x_vals.^10 + c_basis_16(10)*x_vals.^9 + c_basis_16(9)*x_vals.^8 + c_basis_16(8)*x_vals.^7 + c_basis_16(7)*x_vals.^6 + c_basis_16(6)*x_vals.^5 + c_basis_16(5)*x_vals.^4 + c_basis_16(4)*x_vals.^3 + c_basis_16(3)*x_vals.^2 + c_basis_16(2)*x_vals + c_basis_16(1);

% Basis of size 17
A_basis_17 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13 x_samples.^14 x_samples.^15 x_samples.^16];
c_basis_17 = (A_basis_17'*A_basis_17)\(A_basis_17'*data_points');
y_fit_17 = c_basis_17(17)*x_vals.^16 + c_basis_17(16)*x_vals.^15 + c_basis_17(15)*x_vals.^14 + c_basis_17(14)*x_vals.^13 + c_basis_17(13)*x_vals.^12 + c_basis_17(12)*x_vals.^11 + c_basis_17(11)*x_vals.^10 + c_basis_17(10)*x_vals.^9 + c_basis_17(9)*x_vals.^8 + c_basis_17(8)*x_vals.^7 + c_basis_17(7)*x_vals.^6 + c_basis_17(6)*x_vals.^5 + c_basis_17(5)*x_vals.^4 + c_basis_17(4)*x_vals.^3 + c_basis_17(3)*x_vals.^2 + c_basis_17(2)*x_vals + c_basis_17(1);

% Basis of size 18
A_basis_18 = [ones(size(x_samples)) x_samples x_samples.^2 x_samples.^3 x_samples.^4 x_samples.^5 x_samples.^6 x_samples.^7 x_samples.^8 x_samples.^9 x_samples.^10 x_samples.^11 x_samples.^12 x_samples.^13 x_samples.^14 x_samples.^15 x_samples.^16 x_samples.^17];
c_basis_18 = (A_basis_18'*A_basis_18)\(A_basis_18'*data_points');
y_fit_18 = c_basis_18(18)*x_vals.^17 + c_basis_18(17)*x_vals.^16 + c_basis_18(16)*x_vals.^15 + c_basis_18(15)*x_vals.^14 + c_basis_18(14)*x_vals.^13 + c_basis_18(13)*x_vals.^12 + c_basis_18(12)*x_vals.^11 + c_basis_18(11)*x_vals.^10 + c_basis_18(10)*x_vals.^9 + c_basis_18(9)*x_vals.^8 + c_basis_18(8)*x_vals.^7 + c_basis_18(7)*x_vals.^6 + c_basis_18(6)*x_vals.^5 + c_basis_18(5)*x_vals.^4 + c_basis_18(4)*x_vals.^3 + c_basis_18(3)*x_vals.^2 + c_basis_18(2)*x_vals + c_basis_18(1);


%% Plot Results
figure
hold on % Superimpose data set from 2(a)
% plot(x_vals,y_vals,'Color',[0.8,0.2,0.2],'LineWidth',1); grid on;
plot(x_vals,y_fit_2,x_vals,y_fit_5,x_vals,y_fit_8,x_vals,y_fit_10,x_vals,y_fit_14,x_vals,y_fit_18); grid on;legend('k = 2','k=5','k=10','k=14','k=18');
scatter(x_samples,data_points,20,'filled');
hold off


%% Plot the log of the test error versus the polynomial basis dimension k = 1,...,18

% % k = 1
% sse_sum_1 = 0;
% weight_1 = mldivide(x_vals*x_vals',x_vals)*y_fit_1';
% corresponding_y_1 = c_basis_1(1);
% for i = 1:data_points_num
%    sse_sum_1 = sse_sum_1 + ((corresponding_y_1(i) - (weight_1*x_samples(i)).^2));
% end
% mse_1 = sse_sum_1/data_points_num

x_1000_samples = sort(rand(1000,1));
% Calculate Corresponding y-values
data_points_1000 = [];
data_points_1000_num = numel(x_1000_samples);

for i = 1:data_points_1000_num
    data_points_1000 = [data_points_1000 gsig(0.07,x_1000_samples(i))];
end

% k = 2
sse_sum_2 = 0;
weight_2 = mldivide(x_vals*x_vals',x_vals)*y_fit_2';
corresponding_y_2 = c_basis_2(2)*x_1000_samples + c_basis_2(1);
for i = 1:data_points_1000_num
   sse_sum_2 = sse_sum_2 + ((corresponding_y_2(i) - (weight_2*x_1000_samples(i))^2)); 
end
mse_2 = sse_sum_2/data_points_1000_num

% k = 3
sse_sum_3 = 0;
weight_3 = mldivide(x_vals*x_vals',x_vals)*y_fit_3';
corresponding_y_3 = c_basis_3(3)*x_1000_samples.^2 + c_basis_3(2)*x_1000_samples + c_basis_3(1);
for i = 1:data_points_1000_num
   sse_sum_3 = sse_sum_3 + ((corresponding_y_3(i) - (weight_3*x_1000_samples(i))^2));
end
mse_3 = sse_sum_3/data_points_1000_num

% k = 4
sse_sum_4 = 0;
weight_4 = mldivide(x_vals*x_vals',x_vals)*y_fit_4';
corresponding_y_4 = c_basis_4(4)*x_1000_samples.^3 + c_basis_4(3)*x_1000_samples.^2 + c_basis_4(2)*x_1000_samples + c_basis_4(1);
for i = 1:data_points_1000_num
   sse_sum_4 = sse_sum_4 + ((corresponding_y_4(i) - (weight_4*x_1000_samples(i))^2)); 
end
mse_4 = sse_sum_4/data_points_1000_num

% k = 5
sse_sum_5 = 0;
weight_5 = mldivide(x_vals*x_vals',x_vals)*y_fit_5';
corresponding_y_5 = c_basis_5(5)*x_1000_samples.^4 + c_basis_5(4)*x_1000_samples.^3 + c_basis_5(3)*x_1000_samples.^2 + c_basis_5(2)*x_1000_samples + c_basis_5(1);
for i = 1:data_points_1000_num
   sse_sum_5 = sse_sum_5 + ((corresponding_y_5(i) - (weight_5*x_1000_samples(i))^2)); 
end
mse_5 = sse_sum_5/data_points_1000_num

% k = 6
sse_sum_6 = 0;
weight_6 = mldivide(x_vals*x_vals',x_vals)*y_fit_6';
corresponding_y_6 = c_basis_6(6)*x_1000_samples.^5 + c_basis_6(5)*x_1000_samples.^4 + c_basis_6(4)*x_1000_samples.^3 + c_basis_6(3)*x_1000_samples.^2 + c_basis_6(2)*x_1000_samples + c_basis_6(1);
for i = 1:data_points_1000_num
   sse_sum_6 = sse_sum_6 + ((corresponding_y_6(i) - (weight_6*x_1000_samples(i))^2));
end
mse_6 = sse_sum_6/data_points_1000_num

% k = 7
sse_sum_7 = 0;
weight_7 = mldivide(x_vals*x_vals',x_vals)*y_fit_7';
corresponding_y_7 = c_basis_7(7)*x_1000_samples.^6 + c_basis_7(6)*x_1000_samples.^5 + c_basis_7(5)*x_1000_samples.^4 + c_basis_7(4)*x_1000_samples.^3 + c_basis_7(3)*x_1000_samples.^2 + c_basis_7(2)*x_1000_samples + c_basis_7(1);
for i = 1:data_points_1000_num
   sse_sum_7 = sse_sum_7 + ((corresponding_y_7(i) - (weight_7*x_1000_samples(i))^2));
end
mse_7 = sse_sum_7/data_points_1000_num

% k = 8
sse_sum_8 = 0;
weight_8 = mldivide(x_vals*x_vals',x_vals)*y_fit_8';
corresponding_y_8 = c_basis_8(8)*x_1000_samples.^7 + c_basis_8(7)*x_1000_samples.^6 + c_basis_8(6)*x_1000_samples.^5 + c_basis_8(5)*x_1000_samples.^4 + c_basis_8(4)*x_1000_samples.^3 + c_basis_8(3)*x_1000_samples.^2 + c_basis_8(2)*x_1000_samples + c_basis_8(1);
for i = 1:data_points_1000_num
   sse_sum_8 = sse_sum_8 + ((corresponding_y_8(i) - (weight_8*x_1000_samples(i))^2));
end
mse_8 = sse_sum_8/data_points_1000_num

% k = 9
sse_sum_9 = 0;
weight_9 = mldivide(x_vals*x_vals',x_vals)*y_fit_9';
corresponding_y_9 = c_basis_9(9)*x_1000_samples.^8 + c_basis_9(8)*x_1000_samples.^7 + c_basis_9(7)*x_1000_samples.^6 + c_basis_9(6)*x_1000_samples.^5 + c_basis_9(5)*x_1000_samples.^4 + c_basis_9(4)*x_1000_samples.^3 + c_basis_9(3)*x_1000_samples.^2 + c_basis_9(2)*x_1000_samples + c_basis_9(1);
for i = 1:data_points_1000_num
   sse_sum_9 = sse_sum_9 + ((corresponding_y_9(i) - (weight_9*x_1000_samples(i)))^2);
end
mse_9 = sse_sum_9/data_points_1000_num

% k = 10
sse_sum_10 = 0;
weight_10 = mldivide(x_vals*x_vals',x_vals)*y_fit_10';
corresponding_y_10 = c_basis_10(10)*x_1000_samples.^9 + c_basis_10(9)*x_1000_samples.^8 + c_basis_10(8)*x_1000_samples.^7 + c_basis_10(7)*x_1000_samples.^6 + c_basis_10(6)*x_1000_samples.^5 + c_basis_10(5)*x_1000_samples.^4 + c_basis_10(4)*x_1000_samples.^3 + c_basis_10(3)*x_1000_samples.^2 + c_basis_10(2)*x_1000_samples + c_basis_10(1);
for i = 1:data_points_1000_num
   sse_sum_10 = sse_sum_10 + ((corresponding_y_10(i) - (weight_10*x_1000_samples(i)))^2);
end
mse_10 = sse_sum_10/data_points_1000_num

% k = 11
sse_sum_11 = 0;
weight_11 = mldivide(x_vals*x_vals',x_vals)*y_fit_11';
corresponding_y_11 = c_basis_11(11)*x_1000_samples.^10 + c_basis_11(10)*x_1000_samples.^9 + c_basis_11(9)*x_1000_samples.^8 + c_basis_11(8)*x_1000_samples.^7 + c_basis_11(7)*x_1000_samples.^6 + c_basis_11(6)*x_1000_samples.^5 + c_basis_11(5)*x_1000_samples.^4 + c_basis_11(4)*x_1000_samples.^3 + c_basis_11(3)*x_1000_samples.^2 + c_basis_11(2)*x_1000_samples + c_basis_11(1);
for i = 1:data_points_1000_num
   sse_sum_11 = sse_sum_11 + ((corresponding_y_11(i) - (weight_11*x_1000_samples(i)))^2);
end
mse_11 = sse_sum_11/data_points_1000_num

% k = 12
sse_sum_12 = 0;
weight_12 = mldivide(x_vals*x_vals',x_vals)*y_fit_12';
corresponding_y_12 = c_basis_12(12)*x_1000_samples.^11 + c_basis_12(11)*x_1000_samples.^10 + c_basis_12(10)*x_1000_samples.^9 + c_basis_12(9)*x_1000_samples.^8 + c_basis_12(8)*x_1000_samples.^7 + c_basis_12(7)*x_1000_samples.^6 + c_basis_12(6)*x_1000_samples.^5 + c_basis_12(5)*x_1000_samples.^4 + c_basis_12(4)*x_1000_samples.^3 + c_basis_12(3)*x_1000_samples.^2 + c_basis_12(2)*x_1000_samples + c_basis_12(1);
for i = 1:data_points_1000_num
   sse_sum_12 = sse_sum_12 + ((corresponding_y_12(i) - (weight_12*x_1000_samples(i)))^2);
end
mse_12 = sse_sum_12/data_points_1000_num

% k = 13
sse_sum_13 = 0;
weight_13 = mldivide(x_vals*x_vals',x_vals)*y_fit_13';
corresponding_y_13 = c_basis_13(13)*x_1000_samples.^12 + c_basis_13(12)*x_1000_samples.^11 + c_basis_13(11)*x_1000_samples.^10 + c_basis_13(10)*x_1000_samples.^9 + c_basis_13(9)*x_1000_samples.^8 + c_basis_13(8)*x_1000_samples.^7 + c_basis_13(7)*x_1000_samples.^6 + c_basis_13(6)*x_1000_samples.^5 + c_basis_13(5)*x_1000_samples.^4 + c_basis_13(4)*x_1000_samples.^3 + c_basis_13(3)*x_1000_samples.^2 + c_basis_13(2)*x_1000_samples + c_basis_13(1);
for i = 1:data_points_1000_num
   sse_sum_13 = sse_sum_13 + ((corresponding_y_13(i) - (weight_13*x_1000_samples(i)))^2);
end
mse_13 = sse_sum_13/data_points_1000_num

% k = 14
sse_sum_14 = 0;
weight_14 = mldivide(x_vals*x_vals',x_vals)*y_fit_14';
corresponding_y_14 = c_basis_14(14)*x_1000_samples.^13 + c_basis_14(13)*x_1000_samples.^12 + c_basis_14(12)*x_1000_samples.^11 + c_basis_14(11)*x_1000_samples.^10 + c_basis_14(10)*x_1000_samples.^9 + c_basis_14(9)*x_1000_samples.^8 + c_basis_14(8)*x_1000_samples.^7 + c_basis_14(7)*x_1000_samples.^6 + c_basis_14(6)*x_1000_samples.^5 + c_basis_14(5)*x_1000_samples.^4 + c_basis_14(4)*x_1000_samples.^3 + c_basis_14(3)*x_1000_samples.^2 + c_basis_14(2)*x_1000_samples + c_basis_14(1);
for i = 1:data_points_1000_num
   sse_sum_14 = sse_sum_14 + ((corresponding_y_14(i) - (weight_14*x_1000_samples(i)))^2);
end
mse_14 = sse_sum_14/data_points_1000_num

% k = 15
sse_sum_15 = 0;
weight_15 = mldivide(x_vals*x_vals',x_vals)*y_fit_15';
corresponding_y_15 = c_basis_15(15)*x_1000_samples.^14 + c_basis_15(14)*x_1000_samples.^13 + c_basis_15(13)*x_1000_samples.^12 + c_basis_15(12)*x_1000_samples.^11 + c_basis_15(11)*x_1000_samples.^10 + c_basis_15(10)*x_1000_samples.^9 + c_basis_15(9)*x_1000_samples.^8 + c_basis_15(8)*x_1000_samples.^7 + c_basis_15(7)*x_1000_samples.^6 + c_basis_15(6)*x_1000_samples.^5 + c_basis_15(5)*x_1000_samples.^4 + c_basis_15(4)*x_1000_samples.^3 + c_basis_15(3)*x_1000_samples.^2 + c_basis_15(2)*x_1000_samples + c_basis_15(1);
for i = 1:data_points_1000_num
   sse_sum_15 = sse_sum_15 + ((corresponding_y_15(i) - (weight_15*x_1000_samples(i)))^2);
end
mse_15 = sse_sum_15/data_points_1000_num

% k = 16
sse_sum_16 = 0;
weight_16 = mldivide(x_vals*x_vals',x_vals)*y_fit_16';
corresponding_y_16 = c_basis_16(16)*x_1000_samples.^15 + c_basis_16(15)*x_1000_samples.^14 + c_basis_16(14)*x_1000_samples.^13 + c_basis_16(13)*x_1000_samples.^12 + c_basis_16(12)*x_1000_samples.^11 + c_basis_16(11)*x_1000_samples.^10 + c_basis_16(10)*x_1000_samples.^9 + c_basis_16(9)*x_1000_samples.^8 + c_basis_16(8)*x_1000_samples.^7 + c_basis_16(7)*x_1000_samples.^6 + c_basis_16(6)*x_1000_samples.^5 + c_basis_16(5)*x_1000_samples.^4 + c_basis_16(4)*x_1000_samples.^3 + c_basis_16(3)*x_1000_samples.^2 + c_basis_16(2)*x_1000_samples + c_basis_16(1);
for i = 1:data_points_1000_num
   sse_sum_16 = sse_sum_16 + ((corresponding_y_16(i) - (weight_16*x_1000_samples(i)))^2);
end
mse_16 = sse_sum_16/data_points_1000_num

% k = 17
sse_sum_17 = 0;
weight_17 = mldivide(x_vals*x_vals',x_vals)*y_fit_17';
corresponding_y_17 = c_basis_17(17)*x_1000_samples.^16 + c_basis_17(16)*x_1000_samples.^15 + c_basis_17(15)*x_1000_samples.^14 + c_basis_17(14)*x_1000_samples.^13 + c_basis_17(13)*x_1000_samples.^12 + c_basis_17(12)*x_1000_samples.^11 + c_basis_17(11)*x_1000_samples.^10 + c_basis_17(10)*x_1000_samples.^9 + c_basis_17(9)*x_1000_samples.^8 + c_basis_17(8)*x_1000_samples.^7 + c_basis_17(7)*x_1000_samples.^6 + c_basis_17(6)*x_1000_samples.^5 + c_basis_17(5)*x_1000_samples.^4 + c_basis_17(4)*x_1000_samples.^3 + c_basis_17(3)*x_1000_samples.^2 + c_basis_17(2)*x_1000_samples + c_basis_17(1);
for i = 1:data_points_1000_num
   sse_sum_17 = sse_sum_17 + ((corresponding_y_17(i) - (weight_17*x_1000_samples(i)))^2);
end
mse_17 = sse_sum_17/data_points_1000_num

% k = 18
sse_sum_18 = 0;
weight_18 = mldivide(x_vals*x_vals',x_vals)*y_fit_18';
corresponding_y_18 = c_basis_18(18)*x_1000_samples.^17 + c_basis_18(17)*x_1000_samples.^16 + c_basis_18(16)*x_1000_samples.^15 + c_basis_18(15)*x_1000_samples.^14 + c_basis_18(14)*x_1000_samples.^13 + c_basis_18(13)*x_1000_samples.^12 + c_basis_18(12)*x_1000_samples.^11 + c_basis_18(11)*x_1000_samples.^10 + c_basis_18(10)*x_1000_samples.^9 + c_basis_18(9)*x_1000_samples.^8 + c_basis_18(8)*x_1000_samples.^7 + c_basis_18(7)*x_1000_samples.^6 + c_basis_18(6)*x_1000_samples.^5 + c_basis_18(5)*x_1000_samples.^4 + c_basis_18(4)*x_1000_samples.^3 + c_basis_18(3)*x_1000_samples.^2 + c_basis_18(2)*x_1000_samples + c_basis_18(1);
for i = 1:data_points_1000_num
   sse_sum_18 = sse_sum_18 + ((corresponding_y_18(i) - (weight_18*x_1000_samples(i)))^2);
end
mse_18 = sse_sum_18/data_points_1000_num

dimensions = [2:18];
mse_results = [mse_2,mse_3,mse_4,mse_5,mse_6,mse_7,mse_8,mse_9,mse_10,mse_11,mse_12,mse_13,mse_14,mse_15,mse_16,mse_17,mse_18];
log_mse_results = log(mse_results);
figure
plot(dimensions,log_= mse_results);grid on;
