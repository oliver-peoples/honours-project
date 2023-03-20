
%this is an example script showing a use for the hermiteh function
%this script creates the Hermite-Gaussian 4,4 mode and displays it
w0=1; %2x the standard deviation of the gaussian
x0=linspace(-4,4,501);
[X,Y]=meshgrid(x0,x0); %create a grid of width 4 units x 4 units with 501x501 mesh
E0=exp(-(X.^2+Y.^2)/w0^2); %create the gaussian
A=hermiteh(4,sqrt(2)*X/w0).*hermiteh(4,sqrt(2)*Y/w0).*E0; %create the 4,4 hermite gaussian
%display the absoute value of the result
% a=abs(A);
% a=a./max(max(a));
% imshow(a);

gridded_data = griddata(x0, x0, A, X, Y);

imshow(A)