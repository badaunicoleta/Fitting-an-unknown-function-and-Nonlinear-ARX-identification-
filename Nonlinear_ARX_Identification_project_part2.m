%% EXPERIMENTAL DATA
clear
close all
[ident_data, valid_data, ...
 u_ident, y_ident, ...
 u_valid, y_valid, Ts] = get_data('iddata-20.mat');

% identification set
figure,
plot(ident_data);
% axis([0 Ts*size(u_ident, 1) min(u_ident)-1 max(u_ident)+1]);
% title('Identification set');
grid;
Ts*size(u_ident, 1)
min(u_ident)-1
% % validation set
% figure,
% plot(valid_data);
% axis([0 Ts*size(u_valid, 1) min(u_valid)-1 max(y_valid)+1]);
% title('Validation set'); grid;

%% THE PARAMETERS THAT WILL USED FOR TUNING
max_na_nb = 3; max_m = 5;
N = size(u_valid, 1);
combs = get_param_combs(max_na_nb, max_m, N);
results = array2table([combs zeros(size(combs, 1), 4)]);
results.Properties.VariableNames(1:8) = ...
    {'na','nb','m','nr parameters', ...
    'mse ident pred', 'mse valid pred', 'mse ident sim', 'mse valid sim'};

% for testing only three cases
% to test only a few cases, uncomment the following line
% results(4:end, : ) = [];

%% TUNING RESULTS
for i = 1 : size(results, 1)
    na = results{i, "na"};
    nb = results{i, "nb"};
    m = results{i, "m"};
  
    % prediction mode
    theta_pred = get_theta(u_ident, y_ident, m, na, nb);
    
    y_hat_ident_pred = get_y_hat(u_ident, y_ident, theta_pred, m, na, nb);
    y_hat_valid_pred = get_y_hat(u_valid, y_valid, theta_pred, m, na, nb);
    
    results.("mse ident pred")(i) = get_mse(y_ident, y_hat_ident_pred);
    results.("mse valid pred")(i) = get_mse(y_valid, y_hat_valid_pred);
    
    
    % simulation mode
    theta_sim = get_theta(u_ident, y_hat_ident_pred, m, na, nb);
    
    y_hat_ident_sim = get_y_hat(u_ident, y_ident, theta_sim, m, na, nb);
    y_hat_valid_sim = get_y_hat(u_valid, y_valid, theta_sim, m, na, nb);
    
    results.("mse ident sim")(i) = get_mse(y_ident, y_hat_ident_sim);
    results.("mse valid sim")(i) = get_mse(y_valid, y_hat_valid_sim);
    
    sprintf(['Done for na=', num2str(na), ' nb=', num2str(nb), ' m=', num2str(m)])
end

% save the table with results
 writetable(results,'results.csv','Delimiter',',')
 
%% PLOTS FOR THE BEST RESULTS
% uncommet this line to plot directly for the best model in this case
% na = 3; nb = 3; m = 1;

[~, best_index] = min(results.("mse valid sim")(:));
na = results.na(best_index);
nb = results.nb(best_index);
m = results.m(best_index);


% prediction mode
theta_pred = get_theta(u_ident, y_ident, m, na, nb);

y_hat_ident_pred = get_y_hat(u_ident, y_ident, theta_pred, m, na, nb);
y_hat_valid_pred = get_y_hat(u_valid, y_valid, theta_pred, m, na, nb);

mse_ident_pred = get_mse(y_ident, y_hat_ident_pred);
mse_valid_pred = get_mse(y_valid, y_hat_valid_pred);


% simulation mode
theta_sim = get_theta(u_ident, y_hat_ident_pred, m, na, nb);

y_hat_ident_sim = get_y_hat(u_ident, y_ident, theta_sim, m, na, nb);
y_hat_valid_sim = get_y_hat(u_valid, y_valid, theta_sim, m, na, nb);

mse_ident_sim = get_mse(y_ident, y_hat_ident_sim);
mse_valid_sim = get_mse(y_valid, y_hat_valid_sim);

plot_comparison(Ts, m, na, nb, ...
                mse_ident_pred, mse_valid_pred, ...
                mse_ident_sim, mse_valid_sim, ...
                u_ident, y_ident, y_hat_ident_pred, y_hat_ident_sim, ...
                u_valid, y_valid, y_hat_valid_pred, y_hat_valid_sim)
            
%% PLOT THE RELATIONSHIP BETWEEN na, nb, m
% keep only the results with MSE validation simulation <= 1
results_small_mse = results;
for i = 1 : size(results_small_mse, 1)
    if results_small_mse.("mse valid sim")(i) >= 1
        results_small_mse.("mse valid sim")(i) = NaN;
    end
end
idx = find(isnan(results_small_mse.("mse valid sim")));
results_small_mse(idx, :) = [];

% plot "4D" = 3D + color
na = results_small_mse.na(:);
nb = results_small_mse.nb(:);
m = results_small_mse.m(:);
mse_valid_sim = results_small_mse.("mse valid sim")(:);

figure,
markerSize = 100;
scatter3(na, nb, m, markerSize, mse_valid_sim, 'filled');
markerSize = 100;
xlabel('na'); ylabel('nb'); zlabel('m');
cb = colorbar;
cb.Label.String = 'MSE_{simulation}';
title('MSE for validation set depending on na, nb, m');      
%% FUNCTIONS
% Function that retrieves the identification and validation data
function [ident_data, valid_data, ...
          u_ident, y_ident, ...
          u_valid, y_valid, Ts] = get_data(file_name)

    experiment_data = load(file_name);
    ident_data = experiment_data.id;
    valid_data = experiment_data.val;
    
    Ts = experiment_data.id.Ts;
    
    u_ident = experiment_data.id.u;
    y_ident = experiment_data.id.y;
    
    u_valid = experiment_data.val.u;
    y_valid = experiment_data.val.y;
end

% Function that makes the vector of delayed inputs OR outputs
function vector = get_delayed_vector(vect, n, k)
    vector = zeros(1, n);
    
    for i = 1 : n
        if k > i
            vector(i) = vect(k-i);
        else
            vector(i) = 0;
        end
    end
end

% Function that makes the vector of delayed inputs AND outputs
function vector = get_delayed_io(y, u, na, nb, k)
    delayed_inputs = get_delayed_vector(-y, na, k);
    delayed_outputs = get_delayed_vector(u, nb, k);
    vector = [delayed_inputs delayed_outputs];
end

% Function that finds the polynomial of degree m in variables x1...xn 
function poly = get_poly(m, X)
    X_syms = (sym('x',[1 size(X, 2)]));
    
    expr = 1;
    for i = 1 : m
        expr = expr + sum(X_syms)^i;
    end
    
    [~, powers] = coeffs(expr);
    poly = sort(powers); % display this for the syms form
    poly = subs(poly, sym2cell(X_syms), num2cell(X));
    poly = double(poly);
end

% Function that computes the matrix PHI
function phi = get_phi(u, y, m, na, nb)
    N = size(u, 1);
    n = nchoosek(na+nb+m, m);
    phi = zeros(N, n);
    
    for k = 1 : N
        if (k-1 <= 0)
            phi(k, 1:end) = zeros(1, n);
        else
            delayed_io = get_delayed_io(y, u, na, nb, k);
            phi(k, 1:end) = get_poly(m, delayed_io);
            %phi(k, 1:end) = get_poly(m, [-y(k-1) u(k-1)]);
        end
    end
end

% Function that computes the vector theta
function theta = get_theta(u_ident, y_ident, m, na, nb)
    phi = get_phi(u_ident, y_ident, m, na, nb);
    theta = phi\y_ident;
end

% Function that computes y_hat
function y_hat = get_y_hat(u, y, theta, m, na, nb)
    phi = get_phi(u, y, m, na, nb);
    y_hat = phi * theta;
end

% Function that computes the MSE
function mse = get_mse(y, y_hat)
    mse = mean((y - y_hat).^2);
end

% Function that finds all the possible combinations na, nb and m 
function combs = get_param_combs(max_na_nb, max_m, N)
    na = 1 : max_na_nb;
    nb = 1 : max_na_nb;
    m = 1 : max_m;
    
    [ca, cb, cc] = ndgrid(na, nb, m);
    combs = [ca(:), cb(:), cc(:)];
    combs = [combs zeros(size(combs, 1), 1)];
    for i = 1 : size(combs, 1)
        combs(i, end) = nchoosek(combs(i, 1) + combs(i, 2) + combs(i, 3), combs(i, 3)); 
    end
    combs = combs(all(combs(:, end) <= 0.2*N, 2), :);
end

% Plot approximated function for identification and validation data
function plot_comparison(Ts, m, na, nb, ...
                         mse_ident_pred, mse_valid_pred, ...
                         mse_ident_sim, mse_valid_sim, ...
                         u_ident, y_ident, y_hat_ident_pred, y_hat_ident_sim, ...
                         u_valid, y_valid, y_hat_valid_pred, y_hat_valid_sim)
    figure,
    t_ident = linspace(1, size(u_ident, 1)*Ts, size(u_ident, 1));
    plot(t_ident, y_ident); grid; hold on;
    plot(t_ident, y_hat_ident_pred); 
    plot(t_ident, y_hat_ident_sim); hold off;
    xlabel('x'); ylabel('y'); 
    legend('system', 'approximation - prediction mode', 'approximation - simulation mode');
    title('Identification set');
    
    str = [convertCharsToStrings(['MSE_{prediction}=', num2str(mse_ident_pred)]); ...
           convertCharsToStrings(['MSE_{simulation}=', num2str(mse_ident_sim)]); ...
           convertCharsToStrings(['m=', num2str(m), ' na=', num2str(na), ' nb=', num2str(nb)])];
    annotation('textbox',[.6 .2 .1 .1],'String',str,'FitBoxToText','on');
    
    
    figure,
    t_valid = linspace(1, size(u_valid, 1)*Ts, size(u_valid, 1));
    plot(t_valid, y_valid); grid; hold on;
    plot(t_valid, y_hat_valid_pred); 
    plot(t_valid, y_hat_valid_sim); hold off;
    xlabel('x'); ylabel('y'); 
    legend('system', 'approximation - prediction mode', 'approximation - simulation mode');
    title('Validation set');
    
    str = [convertCharsToStrings(['MSE_{prediction}=', num2str(mse_valid_pred)]); ...
           convertCharsToStrings(['MSE_{simulation}=', num2str(mse_valid_sim)]); ...
           convertCharsToStrings(['m=', num2str(m), ' na=', num2str(na), ' nb=', num2str(nb)])];
    annotation('textbox',[.6 .2 .1 .1],'String',str,'FitBoxToText','on');
end
