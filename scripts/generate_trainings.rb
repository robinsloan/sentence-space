require "yaml"

default_batch_size = 32
default_anneal_start = 10000
default_anneal_duration = 10000

session_names = []
train_scripts = []

session_index = 0

[128, 256].each do |z|
  [32].each do |batch_size|
    [64, 128].each do |max_len|
      [0.2, 0.4].each do |alpha|
        ["bigcorpus"].each do |data_source|
          #gpu_string = "CUDA_VISIBLE_DEVICES=#{["0","1"].sample}"

          anneal_start = default_anneal_start * default_batch_size / batch_size
          anneal_duration = default_anneal_duration * default_batch_size / batch_size
          anneal_end = anneal_start + anneal_duration

          session_index += 1
          session_name = "#{session_index}_#{data_source}_z#{z}_alpha#{alpha}_max#{max_len}"
          session_names << session_name
          train_scripts << {:name => session_name, :script => <<-BASH_STRING
cd /home/robin/dev/textproject
python textproject_vae_charlevel.py \\
-session #{session_name} \\
-dataset #{data_source}256.txt \\
-z #{z} \\
-alpha #{alpha} \\
-lr 0.001 \\
-lstm_size 1024 \\
-batch_size #{batch_size} \\
-max_len #{max_len} \\
-anneal_start #{anneal_start} \\
-anneal_end #{anneal_end}
  BASH_STRING
  }
        end
      end
    end
  end
end

File.open("monday_session_names.txt", "w") do |file|
  file.write(session_names.to_yaml)
end

train_scripts.each do |train_script|
  session_name = train_script[:name]
  `mkdir monday_scripts/#{session_name}`
  File.open("monday_scripts/#{session_name}/#{session_name}.sh", "w") do |file|
    file.write(train_script[:script])
  end
end

puts "OK!"
