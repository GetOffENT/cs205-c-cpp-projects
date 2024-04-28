%
% Clear all variables and close all graphs
%

clear all
close all

%
% Load benchmark data
%

run('data/GEMM_BLOCKED_128_128_256.m');  % 文件路径
sizes_IJK = MY_MMult(:, 1);  % 数据大小
times_ns_IJK = MY_MMult(:, 2);  % 执行时间，纳秒
times_ms_IJK = times_ns_IJK / 1e6;  % 转换成毫秒
gflops_IJK = MY_MMult(:, 3);  % GFLOPS

run('data/GEMM_BLOCKED_32_32_32.m'); 
sizes_IKJ = MY_MMult(:, 1); 
times_ns_IKJ = MY_MMult(:, 2); 
times_ms_IKJ = times_ns_IKJ / 1e6;  
gflops_IKJ = MY_MMult(:, 3);  

run('data/GEMM_BLOCKED_64_64_64.m');  
sizes_JIK = MY_MMult(:, 1);  
times_ns_JIK = MY_MMult(:, 2); 
times_ms_JIK = times_ns_JIK / 1e6;  
gflops_JIK = MY_MMult(:, 3); 

run('data/GEMM_BLOCKED_128_128_128.m');  
sizes_JKI = MY_MMult(:, 1);  
times_ns_JKI = MY_MMult(:, 2); 
times_ms_JKI = times_ns_JKI / 1e6;  
gflops_JKI = MY_MMult(:, 3); 

run('data/GEMM_BLOCKED_256_256_256.m');  
sizes_KIJ = MY_MMult(:, 1); 
times_ns_KIJ = MY_MMult(:, 2); 
times_ms_KIJ = times_ns_KIJ / 1e6;  
gflops_KIJ = MY_MMult(:, 3);

run('data/GEMM_BLOCKED_128_64_128.m');  
sizes_128_128_1 = MY_MMult(:, 1);
times_ns_128_128_1 = MY_MMult(:, 2);
times_ms_128_128_1 = times_ns_128_128_1 / 1e6;
gflops_128_128_1 = MY_MMult(:, 3);

run('data/GEMM_BLOCKED_128_256_128.m');
sizes_128_256_128 = MY_MMult(:, 1);
times_ns_128_256_128 = MY_MMult(:, 2);
times_ms_128_256_128 = times_ns_128_256_128 / 1e6;
gflops_128_256_128 = MY_MMult(:, 3);

run('data/GEMM_BLOCKED_64_128_64.m');
sizes_64_128_64 = MY_MMult(:, 1);
times_ns_64_128_64 = MY_MMult(:, 2);
times_ms_64_128_64 = times_ns_64_128_64 / 1e6;
gflops_64_128_64 = MY_MMult(:, 3);

run('data/GEMM_BLOCKED_128_128_64.m');
sizes_KJI = MY_MMult(:, 1);  
times_ns_KJI = MY_MMult(:, 2);
times_ms_KJI = times_ns_KJI / 1e6; 
gflops_KJI = MY_MMult(:, 3); 

run('data/GEMM_BLOCKED_64_128_128.m');
sizes_64_128_128 = MY_MMult(:, 1);
times_ns_64_128_128 = MY_MMult(:, 2);
times_ms_64_128_128 = times_ns_64_128_128 / 1e6;
gflops_64_128_128 = MY_MMult(:, 3);

run('data/BM_GEMM_CBLAS.m');
sizes_CBLAS = MY_MMult(:, 1);
times_ns_CBLAS = MY_MMult(:, 2);
times_ms_CBLAS = times_ns_CBLAS / 1e6;
gflops_CBLAS = MY_MMult(:, 3);

%
% Plot GFLOPS
%

figure;
hold on;
plot(sizes_IJK, gflops_IJK, 'b-o;128\_128\_256;', 'LineWidth', 4, 'MarkerSize', 3); % Increase line width and marker size
plot(sizes_IKJ, gflops_IKJ, 'g-s;32\_32\_32;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_JIK, gflops_JIK, 'r-*;64\_64\_64;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_JKI, gflops_JKI, 'c-^;128\_128\_128;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_KIJ, gflops_KIJ, 'm-<;256\_256\_256;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_128_128_1, gflops_128_128_1, 'm->;128\_64\_128;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_128_256_128, gflops_128_256_128, 'g-o;128\_256\_128;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_64_128_64, gflops_64_128_64, 'b-s;64\_128\_64;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_KJI, gflops_KJI, 'k-p;128\_128\_64;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_64_128_128, gflops_64_128_128, 'r-^;64\_128\_128;', 'LineWidth', 4, 'MarkerSize', 3);
hold off;
legend('Location', 'northwest');
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); % Increase font size for title
set(gca, 'FontSize', 12); % Increase font size for axes ticks
xlim([0 4096]);
ylim([5 32]);
%
% Save the second plot
%

filename = sprintf("gflops_different_block_sizes");
print(filename, '-dpng');
