require "yaml"
require "csv"

require "matplotlib"
Matplotlib::use('Agg')

require "matplotlib/pyplot"
plt = Matplotlib::Pyplot

plt.figure({:figsize => [10,8]})

sessions = YAML.load_file("session_names.txt")

data_array = []
label_array = []

session_ctr = 0

sessions.each do |session|
  data = CSV.read("session/#{session}/log.train.csv")

  time = []
  cost = []
  kld = []

  ctr = 0
  nth_line = 5

  data.each do |line|
    ctr += 1
    if ctr % nth_line == 0
      time << line[0].to_i
      cost << line[1].to_i
      kld << -(line[3].to_i)
    end
  end

  plt.plot(time, cost, '-', {:label => "#{session} cost"})
  plt.plot(time, kld, '--', {:label => "#{session} kld"})

  session_ctr += 1

  if (session_ctr % 5 == 0) || (session_ctr == sessions.length)
    plt.legend({:loc => 'best'})
    plt.ylim([0,200])
    plt.savefig("fig_#{session_ctr}.pdf", {:format => "pdf"})
    `mv fig_#{session_ctr}.pdf ~/dev/titan-share`
    plt.clf
    plt.figure({:figsize => [10,8]})
  end
end


=begin
Prawn::Document.generate "multiplot.pdf", :page_layout => :landscape do
  chart hashed_data, type: :line, line_widths: line_widths
end
=end

puts "Done!"
