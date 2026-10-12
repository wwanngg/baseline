function x_ = FindClosestPoint(x, Flag_dataset)

    if ( nargin == 1 )
      Flag_dataset = 1;
    end

    if(Flag_dataset == 0)
        Interval_gain      =  2;
        Interval_expt      =  150;
        Limit_gain          =  20;
        Lower_limit_expt =  100;
        Upper_limit_expt =  7450;
    elseif(Flag_dataset == 1)
        Interval_gain      =  1;
        Interval_expt      =  3000;
        Limit_gain          =  24;
        Lower_limit_expt =  4000;
        Upper_limit_expt =  67000;
    end

    Gain  = x(1,1);
    Expt  = x(1,2);

    if(mod(Gain,Interval_gain) >= Interval_gain/2)
        Gain = Gain - mod(Gain,Interval_gain) + Interval_gain;
    elseif(mod(Gain,Interval_gain) < Interval_gain/2)
        Gain = Gain - mod(Gain,Interval_gain);
    end

    if(mod(Expt,Interval_expt) >= Interval_expt/2)
        Expt = Expt - mod(Expt,Interval_expt) + Interval_expt + Lower_limit_expt;
    elseif(mod(Expt,Interval_expt) < Interval_expt/2)
        Expt = Expt - mod(Expt,Interval_expt) + Lower_limit_expt;
    end

    if(Gain > Limit_gain)
        Gain = Limit_gain;
    elseif(Gain < 0)
        Gain = 0;
    end

    if(Expt > Upper_limit_expt)
        Expt = Upper_limit_expt;
    elseif(Expt < Lower_limit_expt)
        Expt = Lower_limit_expt;
    end

    x_ = zeros(1,2);
    x_(1,1) = Gain;
    x_(1,2) = Expt;
end
