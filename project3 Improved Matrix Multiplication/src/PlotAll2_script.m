%
% Clear all variables and close all graphs
%

clear all
close all

%
% Load benchmark data
%

run('data/BM_Plain_GEMM_IJK_BLOCKED_128_8.m');  % 文件路径
sizes_IJK = MY_MMult(:, 1);  % 数据大小
times_ns_IJK = MY_MMult(:, 2);  % 执行时间，纳秒
times_ms_IJK = times_ns_IJK / 1e6;  % 转换成毫秒
gflops_IJK = MY_MMult(:, 3);  % GFLOPS

run('data/BM_Plain_GEMM_IJK_BLOCKED_2.m'); 
sizes_IKJ = MY_MMult(:, 1); 
times_ns_IKJ = MY_MMult(:, 2); 
times_ms_IKJ = times_ns_IKJ / 1e6;  
gflops_IKJ = MY_MMult(:, 3); 
%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes_IJK, gflops_IJK, 'b-o;blocked\_128\_8;', 'LineWidth', 4, 'MarkerSize', 3); % Increase line width and marker size
plot(sizes_IKJ, gflops_IKJ, 'g-s;blocked;', 'LineWidth', 4, 'MarkerSize', 3);
hold off;
legend('Location', 'northwest');
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); % Increase font size for title
set(gca, 'FontSize', 12); % Increase font size for axes ticks
%
% Save the second plot
%

filename = sprintf("gflops_script");
print(filename, '-dpng');
