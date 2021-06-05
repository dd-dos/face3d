## Copyright (C) 2013 David Turner
##
## This file is part of Octave.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} clearvars (@var{varargin})
## Clear the listed variables from memory
## @end deftypefn

## Author: David Turner <novalis@novalis.org>

function clearvars (varargin)

  count = 0;
  global_mode = false;
  except_mode = false;
  regexp_mode = false;

  # parse argument list
  for arg = 1:nargin

    curr_arg = varargin{arg};

    if strcmp(curr_arg, "-global")
      global_mode = true;
      continue;
    endif

    if strcmp(curr_arg, "-except")
      except_mode = true;
      regexp_mode = false;
      continue;
    endif

    if strcmp(curr_arg, "-regexp")
      regexp_mode = true;
      continue;
    endif 

    count += 1;
    vars(count).global = global_mode;
    vars(count).except = except_mode;
    vars(count).regexp = regexp_mode;
    vars(count).var_name = curr_arg;

  endfor

  # do we need a wildcard?
  if count == 0 || all([vars.except]) 
    count += 1;
    vars(count).global = global_mode;
    vars(count).except = false;
    vars(count).regexp = false;
    vars(count).var_name = '*';
  endif

  # expand regular expressions
  for c = 1:count

    if vars(c).global == false
        eval_str = 'who(';
    else
        eval_str = 'who("global",';
    endif

    if vars(c).regexp
      eval_str = [eval_str '"-regexp",'];
    endif

    eval_str = [eval_str "'" vars(c).var_name "')"];

    # replace with cell array of matching strings
    vars(c).var_name = evalin('caller',eval_str);

  endfor

  # clear variables one by one (brute force, probably not worth optimizing)
  for c1 = find([vars.except] == false)
    for v1 = 1:numel(vars(c1).var_name)
 
      # look for a match between the clear and except variables
      match = false;

      for c2 = find([vars.except] == true)
        for v2 = 1:numel(vars(c2).var_name)

          match |= isequal(vars(c1).var_name{v1},vars(c2).var_name{v2});

        endfor
      endfor

      # no match found, therefore we clear the variable
      if match == false

        if vars(c).global == false        
          eval_str = ['clear ' vars(c1).var_name{v1}];
        else
          eval_str = ['clear -global ' vars(c1).var_name{v1}];
        endif

        evalin("caller",eval_str);

      endif

    endfor
  endfor

endfunction