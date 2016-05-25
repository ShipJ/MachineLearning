%% Mathematical Programming and Research Methods Coursework 1

%% Part 1 - Gradient Descent

%% Exercise 1(a)

% Create the grid on which to overlay our functions
[X,Y] = meshgrid(linspace(0,5,15),linspace(0,5,15));

figure
% Compute the function for each x and y-value in the range provided
mesh(X,Y,myfunc(X,Y));map=[0,0,0];colormap(map);...
set(gcf,'Color',[1,1,1]);title('Plot of the function f(x,y) = 3(x-3)^2 + (y-2)^2');

%% Exercise 1bii)
% Obtain the sequence of points returned from the gradient descent algorithm
gradPoints = graddesc('fc','dfc',[0,0],0.01,0.1);
 
% Split the sequence of points into x,y and z-values in order to plot them
xVals = gradPoints(1:3:end);
yVals = gradPoints(2:3:end);
zVals = gradPoints(3:3:end);

hold on;
% Plot the generated points on top of function surface
plot3(xVals,yVals,zVals,'Color',[0.2,0.2,0.8],'LineWidth',2); grid on; ...    
    title('Plot of the route traversed by the gradient descent algorithm'); 

%% Exercise 1biii

% Also superimpose the plot of projected points on the xy-plane
plot(xVals,yVals,'Color',[0.8,0.2,0.2],'LineWidth',2); grid on; ... 
    title('Plots of previous exercises superimposed upon each other','FontSize',16); ... 
    legend('f(x,y)','Route traversed','Route projected to xy-plane');
hold off

% One can comment out individual plots to achieve plots shown
% in figures 1,2 and 3, although they look better superimposed.





