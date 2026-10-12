function Test_nelder_mead_AE (Path_UpsampledData)

    close all;

    fprintf ( 1, '\n' );
    fprintf ( 1, 'AE_NELDER_MEAD_TEST:\n' );

    load(Path_UpsampledData, 'path_name', 'dataset_flag');

    n = 2;
    Alpha = 1.7;
    flag_draw = true;
    Flag_Dataset = dataset_flag;

    if(Flag_Dataset == 0)
      Init_Points = zeros(3,2);
      Init_Points(1,1) = 2*(randi(11)-1);
      Init_Points(1,2) = 150*(randi(50)-1) + 100;
    elseif(Flag_Dataset == 1)
      Init_Points = zeros(3,2);
      Init_Points(1,1) = 1*(randi(25)-1);
      Init_Points(1,2) = 3000*(randi(22)-1) + 4000;
    end

    Str_Init_Img = strcat('Param-ISO-',int2str(Init_Points(1,1)), '-ExpT-',int2str(Init_Points(1,2)), '.jpg');
    Path_Init_Img = strcat(path_name,'\',Str_Init_Img);
    Initial_image = imread(Path_Init_Img);

    Intensity = mean2(rgb2gray(Initial_image));

    if(Intensity >= 128)
      h = -1/Alpha*(Intensity/255);
    elseif(Intensity < 128)
      h = Alpha*(1 - Intensity/255);
    end

    Init_Points(2,:) = FindClosestPoint(Init_Points(1,:).*(1 + h*[1,0]), Flag_Dataset);
    Init_Points(3,:) = FindClosestPoint(Init_Points(1,:).*(1 + h*[0,1]), Flag_Dataset);
    fprintf(1,'Initial Param   : Gain : %d, ExpT : %d\n', Init_Points(1,1), Init_Points(1,2) );
    fprintf(1,'Second Param : Gain : %d, ExpT : %d\n', Init_Points(2,1), Init_Points(2,2) );
    fprintf(1,'Third Param    : Gain : %d, ExpT : %d\n', Init_Points(3,1), Init_Points(3,2) );


    fprintf ( 1, '\n' );
    fprintf ( 1, '  Initial data:\n' );
    fprintf ( 1, '\n' );

    for j = 1 : n
    fprintf ( 1, '  ---X(%d)---', j );
    end
    fprintf ( 1, '  ---F(X)---\n' );
    fprintf ( 1, '\n' );
    Metric_our( zeros(1,2) , Path_UpsampledData);

    [ xopt, n_feval ] = nelder_mead_AE ( Init_Points, @Metric_our, flag_draw , Path_UpsampledData);

    fprintf ( 1, '\n' );
    fprintf ( 1, '  Computed optimum after %d evaluations:\n', n_feval );
    fprintf ( 1, '\n' );
    for j = 1 : n
    fprintf ( 1, '  ---X(%d)---', j );
    end
    fprintf ( 1, '  ---F(X)---\n' );
    fprintf ( 1, '\n' );
    for j = 1 : n
    fprintf ( 1, '  %10f', xopt(j) );
    end
    fprintf ( 1, '  %10e\n', Metric_our ( xopt) );

    Result = importdata(strcat(Path_UpsampledData(1:regexp(Path_UpsampledData,'workspace')-1), 'Traj_ExpTGain.txt'));

    Result = Result(4:end,:);
    t = 1:length(Result);
    GainResult = Result(:,1);
    ExpTResult = Result(:,2);

    figure;
    plot(t,GainResult,'LineWidth',2.5);
    set(gca,'FontName', 'Times New Roman','FontSize',17);
    xlabel('Step','fontsize',20, 'fontweight','bold'); ylabel('Gain','fontsize',20, 'fontweight','bold');  grid on;
    tt = [1,5,15,25,35,45];  hold on;
    plot(tt,GainResult(tt),'r*');

    figure;
    plot(t,ExpTResult,'LineWidth',2.5);
    set(gca,'FontName', 'Times New Roman','FontSize',17);
    xlabel('Step','fontsize',20, 'fontweight','bold'); ylabel('ExpT','fontsize',20, 'fontweight','bold');  grid on;
    tt = [1,5,15,25,35,45];  hold on;
    plot(tt,ExpTResult(tt),'r*');

return
end
