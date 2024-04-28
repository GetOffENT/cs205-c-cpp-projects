%
% Clear all variables and close all graphs
%

clear all
close all

%
% Load benchmark data
%

run('data/GEMM_IKJ_OPENMP.m');  % 文件路径
sizes_IJK = MY_MMult(:, 1);  % 数据大小
times_ns_IJK = MY_MMult(:, 2);  % 执行时间，纳秒
times_ms_IJK = times_ns_IJK / 1e6;  % 转换成毫秒
gflops_IJK = MY_MMult(:, 3);  % GFLOPS

run('data/GEMM_BLOCKED_OPENMP.m'); 
sizes_IKJ = MY_MMult(:, 1); 
times_ns_IKJ = MY_MMult(:, 2); 
times_ms_IKJ = times_ns_IKJ / 1e6;  
gflops_IKJ = MY_MMult(:, 3);  

run('data/GEMM_BLOCKED_PACKED_OPENMP.m');  
sizes_JIK = MY_MMult(:, 1);  
times_ns_JIK = MY_MMult(:, 2); 
times_ms_JIK = times_ns_JIK / 1e6;  
gflops_JIK = MY_MMult(:, 3); 

run('data/GEMM_BLOCKED_PACKED_AVX_1_OPENMP.m');  
sizes_JKI = MY_MMult(:, 1);  
times_ns_JKI = MY_MMult(:, 2); 
times_ms_JKI = times_ns_JKI / 1e6;  
gflops_JKI = MY_MMult(:, 3); 

run('data/GEMM_BLOCKED_PACKED_AVX_2_OPENMP.m');  
sizes_KIJ = MY_MMult(:, 1); 
times_ns_KIJ = MY_MMult(:, 2); 
times_ms_KIJ = times_ns_KIJ / 1e6;  
gflops_KIJ = MY_MMult(:, 3);

run('data/BM_Plain_GEMM_KJI.m');
sizes_KJI = MY_MMult(:, 1);  
times_ns_KJI = MY_MMult(:, 2);
times_ms_KJI = times_ns_KJI / 1e6; 
gflops_KJI = MY_MMult(:, 3); 

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
plot(sizes_IJK, gflops_IJK, 'b-o;GEMM\_IKJ\_OPENMP;', 'LineWidth', 4, 'MarkerSize', 3); % Increase line width and marker size
plot(sizes_IKJ, gflops_IKJ, 'g-s;GEMM\_BLOCKED\_OPENMP;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_JIK, gflops_JIK, 'r-*;GEMM\_BLOCKED\_PACKED\_OPENMP;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_JKI, gflops_JKI, 'c-^;GEMM\_BLOCKED\_PACKED\_AVX\_1\_OPENMP;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_KIJ, gflops_KIJ, 'm-<;GEMM\_BLOCKED\_PACKED\_AVX\_2\_OPENMP;', 'LineWidth', 4, 'MarkerSize', 3);
%plot(sizes_KJI, gflops_KJI, 'k-p;Plain\_GEMM\_KJI;', 'LineWidth', 4, 'MarkerSize', 3);
plot(sizes_CBLAS, gflops_CBLAS, 'k-p;OpenBLAS;', 'LineWidth', 4, 'MarkerSize', 3);
hold off;
legend('Location', 'northwest');
xlabel('m = n = k', 'FontSize', 16); % Increase font size for x label
ylabel('GFLOPS', 'FontSize', 16); % Increase font size for y label
title(sprintf('GFLOPS for Various Matrix Sizes'), 'FontSize', 16); % Increase font size for title
set(gca, 'FontSize', 12); % Increase font size for axes ticks
ylim([0 500]);
xlim([0 8192]);
%
% Save the second plot
%

filename = sprintf("gflops_omp");
print(filename, '-dpng');
