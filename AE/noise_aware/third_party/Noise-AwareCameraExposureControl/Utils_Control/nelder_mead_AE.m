function [ x_opt, n_feval ] = nelder_mead_AE (x, function_handle, flag_draw, UpsampledDataResult)

    rho = 1;
    xi  = 2;
    gam = 0.5;
    sig = 0.5;

    tolerance = 1.0E-05;
    max_feval = 50;

    global Img_path;
    global step;
    global Flag_Dataset;

    load(UpsampledDataResult);
    Flag_Dataset = dataset_flag;
    Img_path = path_name;
    step = 1;

    [ temp, n_dim ] = size ( x );

    if ( temp ~= n_dim + 1 )
        fprintf ( 1, '\n' );
        fprintf ( 1, 'NELDER_MEAD - Fatal error!\n' );
        error('  Number of points must be = number of design variables + 1\n');
    end

    if ( flag_draw )
        xp = Xq(1,:);
        yp = Yq(:,1)';
        fp = -Zq;

        [~, idx] = max(Results_Our(:,3));
        GT_value = Results_Our(idx,:);

        figure (2)
        subplot(1,2,1);
        hold on;
        contour(xp,yp,fp,50);
        set(gca,'FontName', 'Times New Roman','FontSize',17);
        xlabel('Gain','fontsize',20, 'fontweight','bold');  ylabel('ExpT','fontsize',20, 'fontweight','bold');
        set(gcf,'units','normalized','outerposition',[0.5 0.5 0.5 0.5])

        plot(x(1,1),x(1,2),'m*','MarkerSize',14,'linewidth',2.5);
        plot(GT_value(1),GT_value(2),'g*','MarkerSize',14,'linewidth',2.5);

        draw_image(x(1,:));
        draw_image(x(2,:));
        draw_image(x(3,:));

        draw_simplex(x);
    end

    [f    ] = evaluate ( x, function_handle );
    n_feval = n_dim + 1;

    [ f, index ] = sort ( f );
    x = x(index,:);

    converged = 0;
    diverged  = 0;

    while ( ~converged && ~diverged )
        x_bar = sum ( x(1:n_dim,:) ) / n_dim;

        x_r   = ( 1 + rho ) * x_bar -  rho * x(n_dim+1,:);
        f_r   = feval(function_handle,x_r);
        n_feval = n_feval + 1;

        if(flag_draw ==1)
            draw_image(x_r);
        end

        if ( f(1) <= f_r && f_r <= f(n_dim) )
            x(n_dim+1,:) = x_r;
            f(n_dim+1  ) = f_r;

            if (flag_draw)
            figure(2);  subplot(1,2,1); title('reflection', 'fontsize',18, 'fontweight','bold')
            end

        elseif ( f_r < f(1) )
            x_e = ( 1 + rho * xi ) * x_bar - rho * xi * x(n_dim+1,:);
            f_e = feval(function_handle,x_e);
            n_feval = n_feval+1;

            if(flag_draw ==1)
                draw_image(x_e);
            end

            if ( f_e < f_r )
                x(n_dim+1,:) = x_e;
                f(n_dim+1  ) = f_e;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('expansion', 'fontsize',18, 'fontweight','bold')
                end
            else
                x(n_dim+1,:) = x_r;
                f(n_dim+1  ) = f_r;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('eventual reflection', 'fontsize',18, 'fontweight','bold')
                end
            end

        elseif ( f(n_dim) <= f_r && f_r < f(n_dim+1) )
            x_c = (1+rho*gam)*x_bar - rho*gam*x(n_dim+1,:);
            f_c = feval(function_handle,x_c); n_feval = n_feval+1;

            if(flag_draw ==1)
                draw_image(x_c);
            end

            if (f_c <= f_r)
                x(n_dim+1,:) = x_c;
                f(n_dim+1  ) = f_c;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('outside contraction', 'fontsize',18, 'fontweight','bold')
                end
            else
               [x,f] = shrink(x,function_handle,sig); n_feval = n_feval+n_dim;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('shrink', 'fontsize',18, 'fontweight','bold')
                end
            end

        else
            x_c = ( 1 - gam ) * x_bar + gam   * x(n_dim+1,:);
            f_c = feval(function_handle,x_c);
            n_feval = n_feval+1;

            if(flag_draw ==1)
                draw_image(x_c);
            end

              if (f_c < f(n_dim+1))
                x(n_dim+1,:) = x_c;
                f(n_dim+1  ) = f_c;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('inside contraction', 'fontsize',18, 'fontweight','bold')
                end
              else
                [x,f] = shrink(x,function_handle,sig); n_feval = n_feval+n_dim;
                if (flag_draw)
                    figure(2); subplot(1,2,1); title('shrink', 'fontsize',18, 'fontweight','bold')
                end
              end
        end


        [ f, index ] = sort ( f );
         x = x(index,:);

        converged = f(n_dim+1)-f(1) < tolerance;

        diverged = ( max_feval < n_feval );

        if (flag_draw)
            draw_simplex(x);
        end
    end

    if ( 0 )
        fprintf('The best point x^* was: %d %d\n',x(1,:));
        fprintf('f(x^*) = %d\n',f(1));
    end

    x_opt = x(1,:);

    figure(2);
    subplot(1,2,1);
    plot(x(1,1),x(1,2),'r*','MarkerSize',14,'linewidth',2.5);
    title('');

    if ( diverged )
        fprintf ( 1, '\n' );
        fprintf ( 1, 'NELDER_MEAD - Warning!\n' );
        fprintf ( 1, '  The maximum number of function evaluations was exceeded\n')
        fprintf ( 1, '  without convergence being achieved.\n' );
    end

    return
end

function f = evaluate ( x, function_handle)

    [ ~, n_dim ] = size ( x );

    f = zeros ( 1, n_dim+1 );

    for i = 1 : n_dim + 1
        f(i) = feval(function_handle,x(i,:));
    end

    return
end

function [ x, f ] = shrink ( x, function_handle, sig )

  [ ~, n_dim ] = size ( x );

  x1 = x(1,:);
  f(1) = feval ( function_handle, x1 );

  for i = 2 : n_dim + 1
    x(i,:) = sig * x(i,:) + ( 1.0 - sig ) * x(1,:);
    f(i) = feval ( function_handle, x(i,:) );
  end

  return
end


function draw_image(x)
    global Img_path step Flag_Dataset;
    step = step + 1;

    figure (2)
    subplot(1,2,1);
    plot(x(1,1), x(1,2), 'k*', 'MarkerSize', 10, 'linewidth', 1.2);

    subplot(1,2,2);
    x_close = FindClosestPoint(x(1,:), Flag_Dataset);
    name_Initial_img = strcat('Param-ISO-', int2str(x_close(1,1)), '-ExpT-', int2str(x_close(1,2)), '.jpg');
    path_Initial_image = strcat(Img_path,'\',name_Initial_img);

    Initial_image = imread(path_Initial_image);
    imshow(Initial_image);
    title_str = strcat('Step : ',num2str(step),' Gain : ', num2str(x(1,1),2) ,'dB,',' ExpT : ', num2str(x(1,2)/1000,4), 'ms' );
    title(title_str, 'fontsize',16, 'fontweight','bold');
    pause(0.1);
end

function draw_simplex(x)
    figure (2)
    subplot(1,2,1);
    plot(x(1:2,1),x(1:2,2),'r')
    plot(x(2:3,1),x(2:3,2),'r')
    plot(x([1 3],1),x([1 3],2),'r')
    pause(0.1);
    plot(x(1:2,1),x(1:2,2),'k')
    plot(x(2:3,1),x(2:3,2),'k')
    plot(x([1 3],1),x([1 3],2),'k')
end
