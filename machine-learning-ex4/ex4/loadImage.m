
function imageVector = loadImage(fileName)

Raw = imread(fileName); % load the black and white jpeg
Raw_float = double(Raw) .* (1/255); % convert the values to floating precision
Raw_vector = Raw_float(:)';% To turn the matrix into a vector

displayData(Raw_vector); % for viewing

imageVector = Raw_vector;

% predict(Theta1, Theta2, imageVector) % to predict 
