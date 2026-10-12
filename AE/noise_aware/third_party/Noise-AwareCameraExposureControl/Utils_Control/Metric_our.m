function f = Metric_our ( x , Path_UpsampledData)

  if ( length ( x ) ~= 2 )
    error ( 'Error: function expects a two dimensional input\n' );
  end

    persistent Xq Yq Zq Interval_Up_Gain Interval_Up_ExpT result_txt;
    if ( nargin == 2 )
        load(Path_UpsampledData,'Xq','Yq','Zq', 'interval_ExpT','interval_dB');
        Interval_Up_Gain = interval_dB;
        Interval_Up_ExpT = interval_ExpT;
        result_txt = strcat(Path_UpsampledData(1:regexp(Path_UpsampledData,'workspace')-1), 'Traj_ExpTGain.txt');
        fileID = fopen(result_txt,'w');
        fclose(fileID);
    else
        fileID = fopen(result_txt,'a');
        fprintf(fileID , '%4.5f %4.5f\n' , x(1) , x(2));
        fclose(fileID);
    end

    Gain = x(1);
    Expt = x(2);

    if(mod(Gain, Interval_Up_Gain) >= Interval_Up_Gain/2)
        Approxi_Gain = Gain - mod(Gain, Interval_Up_Gain) + Interval_Up_Gain;
    elseif(mod(Gain, Interval_Up_Gain) < Interval_Up_Gain/2)
        Approxi_Gain = Gain - mod(Gain, Interval_Up_Gain);
    end

    if(mod(Expt, Interval_Up_ExpT) >= Interval_Up_ExpT/2)
        Approxi_Expt = Expt - mod(Expt, Interval_Up_ExpT) + Interval_Up_ExpT;
    elseif(mod(Expt, Interval_Up_ExpT) < Interval_Up_ExpT/2)
       Approxi_Expt = Expt - mod(Expt, Interval_Up_ExpT);
    end

    Gain_index = find(abs(Xq(1,:) - Approxi_Gain) < 0.001);
    ExpT_index = find(abs(Yq(:,1) - Approxi_Expt) < 0.001);

    f =  -Zq(ExpT_index,Gain_index);

    if(isnan(f))
        f = 10^10;
    elseif(isempty(f))
        f = 10^10;
    end

      fprintf ( 1, '  %9.3f',  Approxi_Gain );
      fprintf ( 1, '  %9.3f',  Approxi_Expt );
      fprintf ( 1, '  %9.3e\n', f );

    return
end
