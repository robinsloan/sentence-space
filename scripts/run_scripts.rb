require "yaml"

sessions = YAML.load_file("monday_session_names.txt")

sessions.each do |session|
  puts "Running #{session}..."
  `tmux new-session -d -s #{session} 'bash monday_scripts/#{session}/#{session}.sh'`
  puts "(Sleeping for 10 seconds...)"
  sleep(10)
end

puts "Done!"
