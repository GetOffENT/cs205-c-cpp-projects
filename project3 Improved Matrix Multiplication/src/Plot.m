%
% Clear all variables and close all graphs
%

clear all
close all

%
% Load benchmark data
%

% 确保这里的路径正确指向您的 .m 文件
run('data/BM_Plain_GEMM_IJK.m');  % 修改为您的文件实际路径

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  % 数据大小
times_ns = MY_MMult(:, 2);  % 执行时间，纳秒
times_ms = times_ns / 1e6;  % 转换成毫秒
gflops = MY_MMult(:, 3);  % GFLOPS
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_IJK;', 'LineWidth', 4, 'MarkerSize', 3); % Increase line width and marker size
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); % Increase font size for title
set(gca, 'FontSize', 12); % Increase font size for axes ticks
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');




run('data/BM_Plain_GEMM_IKJ.m'); 

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  
times_ns = MY_MMult(:, 2); 
times_ms = times_ns / 1e6;  
gflops = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_IKJ;', 'LineWidth', 4, 'MarkerSize', 3); 
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); 
set(gca, 'FontSize', 12); 
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');




run('data/BM_Plain_GEMM_JIK.m'); 

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  
times_ns = MY_MMult(:, 2); 
times_ms = times_ns / 1e6;  
gflops = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_JIK;', 'LineWidth', 4, 'MarkerSize', 3); 
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); 
set(gca, 'FontSize', 12); 
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');




run('data/BM_Plain_GEMM_JKI.m'); 

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  
times_ns = MY_MMult(:, 2); 
times_ms = times_ns / 1e6;  
gflops = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_JKI;', 'LineWidth', 4, 'MarkerSize', 3); 
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); 
set(gca, 'FontSize', 12); 
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');





run('data/BM_Plain_GEMM_KIJ.m'); 

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  
times_ns = MY_MMult(:, 2); 
times_ms = times_ns / 1e6;  
gflops = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_KIJ;', 'LineWidth', 4, 'MarkerSize', 3); 
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); 
set(gca, 'FontSize', 12); 
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');



run('data/BM_Plain_GEMM_KJI.m'); 

%
% Extract data for plotting
%

sizes = MY_MMult(:, 1);  
times_ns = MY_MMult(:, 2); 
times_ms = times_ns / 1e6;  
gflops = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes, gflops, 'r-*;Plain\_GEMM\_KJI;', 'LineWidth', 4, 'MarkerSize', 3); 
hold off;
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); 
set(gca, 'FontSize', 12); 
xlim([0 1024]);
%
% Save the second plot
%

filename = sprintf("gflops_%s", version);
print(filename, '-dpng');
