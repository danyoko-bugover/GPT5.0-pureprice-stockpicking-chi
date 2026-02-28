# clean_data_folders_fixed.py
# 修正版：直接用 input() 等待使用者確認，不用複雜非阻塞

import os
import shutil

TARGETS = [
    "data",
    "data_processed",
    "input",
    "output",
    "mapping.txt"
]

def main():
    print("=== 清理腳本（修正版） ===")
    print(f"目前目錄: {os.getcwd()}\n")
    
    if not os.path.exists("save_csi300_prices.py"):
        print("警告：沒找到 save_csi300_prices.py，看起來不在專案根目錄？")
        ans = input("仍要繼續？(y/n): ").strip().lower()
        if ans != 'y':
            print("取消。")
            return
    
    to_delete = [t for t in TARGETS if os.path.exists(t)]
    
    if not to_delete:
        print("沒有任何目標需要刪除。結束。")
        return
    
    print("將刪除以下項目：")
    for item in to_delete:
        if os.path.isdir(item):
            print(f"  - {item}/")
        else:
            print(f"  - {item}")
    
    confirm = input("\n確定要刪除？(y/n): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("取消。")
        return
    
    for item in to_delete:
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"已刪除: {item}/")
            else:
                os.remove(item)
                print(f"已刪除: {item}")
        except Exception as e:
            print(f"刪除 {item} 失敗: {e}")
    
    print("\n清理完成。")
    print("建議執行：git add -u && git commit -m 'Clean data folders'")

if __name__ == "__main__":
    main()