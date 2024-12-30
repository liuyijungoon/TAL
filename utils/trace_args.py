import inspect
from functools import wraps
import os

def track_args_usage(obj, target_path="FMFP"):  # 添加target_path参数
    def is_target_file(file_path):
        return target_path in file_path
    
    def filter_locations(locations):
        return {loc for loc in locations if is_target_file(loc.split(':')[0])}

    # 如果是类
    if isinstance(obj, type):
        original_init = obj.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            self.accessed_fields = {}
            
            original_getattr = type(self).__getattribute__
            
            def tracked_getattr(obj, name):
                if not name.startswith('_') and name != 'accessed_fields':
                    frame = inspect.currentframe()
                    caller_frame = frame.f_back
                    caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
                    
                    # 只记录目标路径下的访问
                    if is_target_file(caller_frame.f_code.co_filename):
                        if name not in obj.accessed_fields:
                            obj.accessed_fields[name] = set()
                        obj.accessed_fields[name].add(caller_info)
                    
                    del frame
                    del caller_frame
                return original_getattr(obj, name)
            
            type(self).__getattribute__ = tracked_getattr
            
            # 执行原始__init__
            original_init(self, *args, **kwargs)
            
            # 打印使用情况
            print_usage(self)
            
            # 恢复原始getattr
            type(self).__getattribute__ = original_getattr
        
        obj.__init__ = new_init
        return obj
    
    # 如果是函数
    else:
        @wraps(obj)
        def wrapper(*args, **kwargs):
            if len(args) > 0 and hasattr(args[0], '__dict__'):
                args_obj = args[0]
                args_obj.accessed_fields = {}
                
                original_getattr = type(args_obj).__getattribute__
                
                def tracked_getattr(self, name):
                    if not name.startswith('_') and name != 'accessed_fields':
                        frame = inspect.currentframe()
                        caller_frame = frame.f_back
                        caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
                        
                        # 只记录目标路径下的访问
                        if is_target_file(caller_frame.f_code.co_filename):
                            if name not in self.accessed_fields:
                                self.accessed_fields[name] = set()
                            self.accessed_fields[name].add(caller_info)
                        
                        del frame
                        del caller_frame
                    return original_getattr(self, name)
                
                type(args_obj).__getattribute__ = tracked_getattr
                
                # 执行原始函数
                result = obj(*args, **kwargs)
                
                # 打印使用情况
                print_usage(args_obj)
                
                # 恢复原始getattr
                type(args_obj).__getattribute__ = original_getattr
                
                return result
            else:
                return obj(*args, **kwargs)
        
        return wrapper

def print_usage(obj):
    # 过滤掉没有使用记录的参数
    filtered_fields = {
        name: locations 
        for name, locations in obj.accessed_fields.items() 
        if locations  # 只保留有访问记录的参数
    }
    
    if filtered_fields:
        print("\n\nArguments usage tracking (in FMFP):")
        print("=" * 50)
        
        # 打印已使用的参数
        print("\nUsed arguments:")
        for arg_name, locations in filtered_fields.items():
            print(f"\n{arg_name}: {getattr(obj, arg_name)}")
            print("Used in:")
            for loc in locations:
                file, line = loc.split(':')
                print(f"  - Line {line} in {os.path.basename(file)}")
    
        # 打印未使用的参数
        all_args = {name for name in dir(obj) 
                   if not name.startswith('_') and 
                   name != 'accessed_fields' and 
                   not callable(getattr(obj, name))}  # 排除方法
        used_args = set(filtered_fields.keys())
        unused_args = all_args - used_args
        if unused_args:
            print("\nUnused arguments:")
            for arg in unused_args:
                print(f"- {arg}: {getattr(obj, arg)}")
