SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND p.PostTypeId = 1
  AND p.ViewCount <= 4159
  AND p.CommentCount >= 0
  AND p.CommentCount <= 12
  AND ph.PostHistoryTypeId = 2
  AND v.VoteTypeId = 2
  AND u.Views <= 86
  AND u.DownVotes >= 0
  AND u.DownVotes <= 1
  AND u.UpVotes <= 6;
