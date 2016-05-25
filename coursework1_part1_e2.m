%% Mathematical Programming and Research Methods Coursework 1

%% Part 1 - Gradient Descent

%% Exercise 2(b)

% Create the matrix of coefficients
A = [1,-1;1,1;1,2]
% Create the vector of solutions
b = [1;1;3]
% Set an initial guess
guess = [0;0];
% Set a step size, there are 4 to illustrate the difference
step = 0.1;
step2 = 0.075;
step3 = 0.05;
step4 = 0.01;
% Set a tolerance, i.e. the value that the gradient must be smaller than in
% order to converge successfully. 
tol = 0.0001;

% Store the guesses and gradients traversed over the algorithm iteration
% The 4 different sets display the points traversed over the different
% step sizes
guesses = mydescent(A, b, guess, step, tol);
guesses2 = mydescent(A, b, guess, step2, tol);
guesses3 = mydescent(A, b, guess, step3, tol);
guesses4 = mydescent(A, b, guess, step4, tol);

% Print the solution to the equation using the first provided step size
solution = [guesses(1,end),guesses(2,end)]

%% Exercise 2(c)

% Plots of Gradient Descent, for each step size
xVals = guesses(1,:);
yVals = guesses(2,:);

xVals2 = guesses2(1,:);
yVals2 = guesses2(2,:);

xVals3 = guesses3(1,:);
yVals3 = guesses3(2,:);

xVals4 = guesses4(1,:);
yVals4 = guesses4(2,:);

figure
hold on
plot(xVals,yVals,'-',xVals2,yVals2,'-',xVals3,yVals3,'-',xVals4,yVals4,'-','LineWidth',1.5);legend('Step size:0.1','Step size:0.075','Step size:0.05','Step size: 0.01');grid on; ...
    title('Convergence to the least squares solution given by my gradient descent algorithm, for different step sizes','FontSize',16);
    set(gcf,'Color',[1,1,1])
hold off
