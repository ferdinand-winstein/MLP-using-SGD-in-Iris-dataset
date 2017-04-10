clear

% load data iris
load irisdataset.txt
data = irisdataset;
N = size(data,1);
%load data target
load iristarget.txt
target = iristarget;

%input bias
%a = input('bias = ');
a = 0.65;

% random weight -.5 < w < .5
w_min = -.5;
w_max = 0.5;
w = w_min + rand(N,12) * (w_max-w_min);
w_awal = w;

% init;
epoch  = 1;
cek = false;
net_h = zeros(N,2);
out_h = zeros(N,2);

net_o = zeros(N,2);
out_o = zeros(N,2);


while cek == false;
    for i = 1:N;
        net_h(i,1) = w(i,1) * data(i,1) + w(i,3) * data(i,2) + w(i,5) * data(i,3) + w(i,7) * data(i,4);
        out_h(i,1) = 1 / (1+exp(-net_h(i,1)));
        net_h(i,2) = w(i,2) * data(i,1) + w(i,4) * data(i,2) + w(i,6) * data(i,3) + w(i,8) * data(i,4);
        out_h(i,2) = 1 / (1+exp(-net_h(i,2)));
        
        net_o(i,1) =  w(i,9) * out_h(i,1) + w(i,11) * out_h(i,2);
        out_o(i,1) = 1 / (1+exp(-net_o(i,1)));
        net_o(i,2) = w(i,10) * out_h(i,1) + w(i,12) * out_h(i,2);
        out_o(i,2) = 1 / (1+exp(-net_o(i,2)));
    end
    output = round(out_o);
    
    % jika output (out_o) sudah sama dengan target, tidak perlu dilanjutkan lagi
    ck = true;
    for i = 1:N;
        if target(i,1) == output(i,1) 
            if target(i,2) == output(i,2)
            end
        else
            ck = false;
        end
    end
            
    %buat weight baru;
    n_w = w;
    Etot = zeros(N,2);
    OutO = zeros(N,2);
    OutH = zeros(N,2);

    for i = 1:N;
        Etot(i,1) = -(target(i,1) - out_o(i,1));
        Etot(i,2) = -(target(i,2) - out_o(i,2));
        OutO(i,1) = out_o(i,1)*(1-out_o(i,1));
        OutO(i,2) = out_o(i,2)*(1-out_o(i,2));
        OutH(i,1) = out_h(i,1)*(1-out_h(i,1));
        OutH(i,2) = out_h(i,2)*(1-out_h(i,2));

        %weight baru
        n_w(i,1) = w(i,1)-a*(((Etot(i,1)*OutO(i,1)*w(i,1))+(Etot(i,2)*OutO(i,2)*w(i,1)))*OutH(i,1)*data(i,1));
        n_w(i,2) = w(i,2)-a*(((Etot(i,2)*OutO(i,1)*w(i,2))+(Etot(i,2)*OutO(i,2)*w(i,2)))*OutH(i,2)*data(i,1));
        n_w(i,3) = w(i,3)-a*(((Etot(i,1)*OutO(i,1)*w(i,3))+(Etot(i,2)*OutO(i,2)*w(i,3)))*OutH(i,1)*data(i,2));
        n_w(i,4) = w(i,4)-a*(((Etot(i,2)*OutO(i,1)*w(i,4))+(Etot(i,2)*OutO(i,2)*w(i,4)))*OutH(i,2)*data(i,2));
        n_w(i,5) = w(i,5)-a*(((Etot(i,1)*OutO(i,1)*w(i,5))+(Etot(i,2)*OutO(i,2)*w(i,5)))*OutH(i,1)*data(i,3));
        n_w(i,6) = w(i,6)-a*(((Etot(i,2)*OutO(i,1)*w(i,6))+(Etot(i,2)*OutO(i,2)*w(i,6)))*OutH(i,2)*data(i,3));
        n_w(i,7) = w(i,7)-a*(((Etot(i,1)*OutO(i,1)*w(i,7))+(Etot(i,2)*OutO(i,2)*w(i,7)))*OutH(i,1)*data(i,4));
        n_w(i,8) = w(i,8)-a*(((Etot(i,2)*OutO(i,1)*w(i,8))+(Etot(i,2)*OutO(i,2)*w(i,8)))*OutH(i,2)*data(i,4));
        
        n_w(i,9) = w(i,9)-a*(Etot(i,1)*OutO(i,1)*out_h(i,1));
        n_w(i,10) = w(i,10)-a*(Etot(i,2)*OutO(i,2)*out_h(i,1));
        n_w(i,11) = w(i,11)-a*(Etot(i,1)*OutO(i,1)*out_h(i,2));
        n_w(i,12) = w(i,12)-a*(Etot(i,2)*OutO(i,2)*out_h(i,2));
    end
    w = n_w;
        
    if ck == true; 
        cek = true;
    else
        cek = false;
        epoch = epoch + 1;
    end
end

dlmwrite('weight_awal.txt',w_awal);
dlmwrite('weight_akhir.txt',w);

dlmwrite('weight1.txt',w(1:40,:));
dlmwrite('weight2.txt',w(41:80,:));
dlmwrite('weight3.txt',w(81:120,:));
